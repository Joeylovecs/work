"""OpenAI-compatible client with retries, bounded cache and secret-safe metadata."""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
from typing import Any, Dict, List, Optional

class APIConfigError(RuntimeError): pass
class ParateraClient:
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
            data=json.loads(path.read_text(encoding="utf-8")); data["cached"]=True; return data
        started=time.perf_counter(); last=None
        for attempt in range(self.max_retries):
            self.calls+=1
            try:
                response=self.client.chat.completions.create(model=self.model,messages=messages,temperature=temperature,max_tokens=max_tokens)
                usage=getattr(response,"usage",None)
                usage_dict={k:getattr(usage,k,None) for k in ("prompt_tokens","completion_tokens","total_tokens")} if usage else {}
                data={"text":response.choices[0].message.content or "","usage":usage_dict,"latency_seconds":time.perf_counter()-started,"api_calls":1,"cached":False,"model":self.model,"prompt_hash":key}
                if path: path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
                return data
            except Exception as exc:
                last=f"{type(exc).__name__}: {exc}"[:500]
                if attempt+1<self.max_retries: time.sleep(2**attempt)
        return {"text":"","usage":{},"latency_seconds":time.perf_counter()-started,"api_calls":self.max_retries,"cached":False,"model":self.model,"prompt_hash":key,"error":last}
