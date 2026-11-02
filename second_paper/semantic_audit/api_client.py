"""OpenAI-compatible client with retries, bounded cache and secret-safe metadata."""
from __future__ import annotations
import hashlib, json, os, time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

class APIConfigError(RuntimeError): pass
class ModelOutputError(RuntimeError): pass
class ParateraClient:
    heartbeat_seconds = 10.0

    def progress(self, message: str) -> None:
        """Report status without logging prompts, headers or credentials."""
        context = getattr(self, "progress_context", "API")
        line = f"[{time.strftime('%H:%M:%S')}] [{context}] {message}"
        print(line, flush=True)
        cache_dir = getattr(self, "cache_dir", None)
        if cache_dir:
            try:
                with (cache_dir.parent / "progress.log").open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError:
                # A status log must not change whether a model result is accepted.
                pass

    @contextmanager
    def request_progress(self, attempt: int):
        stopped = threading.Event()
        started = time.perf_counter()

        def heartbeat():
            while not stopped.wait(self.heartbeat_seconds):
                self.progress(f"等待API返回 已等{time.perf_counter() - started:.0f}s "
                              f"外层尝试={attempt}/{self.max_retries}")

        self.progress(f"发送API请求 model={self.model} 外层尝试={attempt}/{self.max_retries} "
                      f"timeout={self.timeout:g}s")
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stopped.set()
            worker.join()

    def __init__(self, cache_dir: Optional[str]=None, timeout: float=120.0, max_retries: int=3):
        self.model=os.getenv("MODEL_ID","DeepSeek-V3.2")
        self.base_url=os.getenv("PARATERA_BASE_URL","https://ai.paratera.com/v1/")
        self.timeout=timeout; self.max_retries=int(os.getenv("PARATERA_MAX_RETRIES", str(max_retries))); self.calls=0
        self.cache_dir=Path(cache_dir) if cache_dir else None
        if self.cache_dir: self.cache_dir.mkdir(parents=True,exist_ok=True)
        key=os.getenv("PARATERA_API_KEY")
        if not key:
            secret_file=Path(__file__).resolve().parents[1]/".secrets"/"paratera.env"
            if secret_file.exists():
                for line in secret_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("PARATERA_API_KEY="):
                        key=line.split("=",1)[1].strip(); break
        if not key: raise APIConfigError("PARATERA_API_KEY is not configured; API experiment not started")
        from openai import OpenAI
        self.client=OpenAI(api_key=key,base_url=self.base_url,timeout=timeout,max_retries=0)
    @property
    def thinking_request(self):
        mode = os.getenv("PARATERA_GLM_THINKING", "default") if self.model.lower() == "glm-5.3" else "default"
        if mode not in ("default", "enabled", "disabled"):
            raise APIConfigError("PARATERA_GLM_THINKING must be default, enabled or disabled")
        return mode

    def _key(self, messages, temperature, max_tokens):
        signature = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if self.thinking_request != "default":
            signature["thinking"] = self.thinking_request
        raw = json.dumps(signature, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()
    def _completion(self, messages, temperature, max_tokens):
        params = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if self.model.lower() != "glm-5.3":
            result = self.client.chat.completions.create(**params)
            choice = result.choices[0] if result.choices else None
            content = choice.message.content if choice else ""
            return (content or "").strip(), getattr(choice, "finish_reason", None), getattr(result, "usage", None)

        # Streaming prevents long reasoning periods from leaving the gateway idle.
        if self.thinking_request != "default":
            params["extra_body"] = {"thinking": {"type": self.thinking_request}}
        parts, finish, usage = [], None, None
        with self.client.chat.completions.create(**params, stream=True, stream_options={"include_usage": True}) as stream:
            for chunk in stream:
                usage = getattr(chunk, "usage", None) or usage
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        parts.append(choice.delta.content)
                    finish = choice.finish_reason or finish
        if not finish:
            raise ConnectionError("Model stream ended without a finish marker")
        return "".join(parts).strip(), finish, usage

    def chat(self, messages: List[Dict[str,str]], temperature: float=0.0, max_tokens: int=1200, cache: bool=True) -> Dict[str,Any]:
        key=self._key(messages,temperature,max_tokens); path=self.cache_dir/(key+".json") if self.cache_dir else None
        if cache and path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if (data.get("text") or "").strip() and data.get("finish_reason") != "length":
                self.progress("命中本次实验响应缓存")
                data["cached"] = True
                return data
        started = time.perf_counter()
        last = "No attempts configured"
        usage_total = {}
        requested_tokens = max_tokens
        for attempt in range(self.max_retries):
            self.calls+=1
            try:
                with self.request_progress(attempt + 1):
                    text, finish, usage = self._completion(messages, temperature, max_tokens)
                self.progress(f"API已返回 累计等待={time.perf_counter() - started:.1f}s")
                usage_dict={k:getattr(usage,k,None) for k in ("prompt_tokens","completion_tokens","total_tokens")} if usage else {}
                for name, count in usage_dict.items():
                    usage_total[name] = usage_total.get(name, 0) + (count or 0)
                exhausted = finish == "length" or (not text and usage_dict.get("completion_tokens") == max_tokens)
                if exhausted or not text:
                    detail = (f"model={self.model} finish={finish} max_tokens={max_tokens} "
                              f"completion_tokens={usage_dict.get('completion_tokens')} content_chars={len(text)}")
                    if (exhausted and self.model.lower() == "glm-5.3"
                            and max_tokens == requested_tokens and max_tokens < 8192
                            and attempt + 1 < self.max_retries):
                        max_tokens = min(8192, max(4096, max_tokens * 2))
                        self.progress(f"Truncated response; retry with max_tokens={max_tokens}. {detail}")
                        continue
                    self.progress(f"Unusable model response: {detail}")
                    raise ModelOutputError(f"Unusable model response: {detail}")
                data={"text":text,"usage":usage_total,"finish_reason":finish,"max_tokens":max_tokens,
                      "latency_seconds":time.perf_counter()-started,"api_calls":attempt+1,
                      "cached":False,"model":self.model,"prompt_hash":key,"thinking_request":self.thinking_request}
                if cache and path: path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
                return data
            except ModelOutputError:
                raise
            except Exception as exc:
                causes = []
                cause = exc
                while cause is not None and len(causes) < 5:
                    causes.append(type(cause).__name__)
                    cause = cause.__cause__
                last = f"{' -> '.join(causes)} status={getattr(exc, 'status_code', None)}"
                self.progress(f"API失败 type={type(exc).__name__} status={getattr(exc, 'status_code', None)} "
                              f"外层尝试={attempt + 1}/{self.max_retries}")
                if attempt+1<self.max_retries:
                    self.progress(f"{2**attempt}s后重试")
                if attempt+1<self.max_retries: time.sleep(2**attempt)
        self.progress("API外层重试耗尽；本条没有获取到模型回复，检查请求错误后再判断准确率")
        self.progress(f"Request failed: {last}")
        raise RuntimeError(f"Model request failed after {self.max_retries} attempts: {last}") from None
