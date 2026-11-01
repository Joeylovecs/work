"""OpenAI-compatible client with retries, bounded cache and secret-safe metadata."""
from __future__ import annotations
import hashlib, json, os, time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

class APIConfigError(RuntimeError): pass
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
                              f"外层尝试={attempt}/{self.max_retries}；SDK可能正在内部重试")

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
        self.client=OpenAI(api_key=key,base_url=self.base_url,timeout=timeout)
    def _key(self, messages, temperature, max_tokens):
        raw=json.dumps({"model":self.model,"messages":messages,"temperature":temperature,"max_tokens":max_tokens},sort_keys=True,ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()
    def chat(self, messages: List[Dict[str,str]], temperature: float=0.0, max_tokens: int=1200, cache: bool=True) -> Dict[str,Any]:
        key=self._key(messages,temperature,max_tokens); path=self.cache_dir/(key+".json") if self.cache_dir else None
        if cache and path and path.exists():
            self.progress("命中本次实验响应缓存")
            data=json.loads(path.read_text(encoding="utf-8")); data["cached"]=True; return data
        started=time.perf_counter(); last=None
        for attempt in range(self.max_retries):
            self.calls+=1
            try:
                with self.request_progress(attempt + 1):
                    response=self.client.chat.completions.create(model=self.model,messages=messages,temperature=temperature,max_tokens=max_tokens)
                self.progress(f"API已返回 累计等待={time.perf_counter() - started:.1f}s")
                usage=getattr(response,"usage",None)
                usage_dict={k:getattr(usage,k,None) for k in ("prompt_tokens","completion_tokens","total_tokens")} if usage else {}
                data={"text":response.choices[0].message.content or "","usage":usage_dict,"latency_seconds":time.perf_counter()-started,"api_calls":1,"cached":False,"model":self.model,"prompt_hash":key}
                if path: path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
                return data
            except Exception as exc:
                last=f"{type(exc).__name__}: {exc}"[:500]
                self.progress(f"API失败 type={type(exc).__name__} status={getattr(exc, 'status_code', None)} "
                              f"外层尝试={attempt + 1}/{self.max_retries}")
                if attempt+1<self.max_retries:
                    self.progress(f"{2**attempt}s后重试")
                if attempt+1<self.max_retries: time.sleep(2**attempt)
        self.progress("API外层重试耗尽；本条没有获取到模型回复，检查请求错误后再判断准确率")
        return {"text":"","usage":{},"latency_seconds":time.perf_counter()-started,"api_calls":self.max_retries,"cached":False,"model":self.model,"prompt_hash":key,"error":last}
