import openai
from openai import OpenAI
import os
import httpx
import time
import tiktoken
from typing import Optional, Union
import torch
import threading
import queue


class Model:
    def __init__(self, model_name: str, provider: str = 'openai', device: Optional[str] = None):
        self.model_name = model_name
        self.provider = provider  # 'openai' or 'huggingface'
        if provider == 'huggingface':
            # 推迟导入 transformers，避免在 openai 模式下强依赖
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
            # 优先支持本地/远程 HF 模型（如 Qwen），自动上 GPU
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, use_fast=True)
            # 记录是否存在聊天模板
            self._has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and getattr(
                self.tokenizer, "chat_template", None)
            # 处理缺省的 pad_token
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            # 优先使用 bfloat16（Qwen3 推荐），不支持 bf16 时回退到 fp16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            if torch.cuda.is_available():
                # 清理GPU缓存，避免碎片化
                torch.cuda.empty_cache()

            # 根据模型类型自动选择 attention 实现：
            _model_name_lower = model_name.lower()
            if "qwen" in _model_name_lower:
                _attn_impl = "sdpa"
                if torch.cuda.is_available() and os.environ.get("QWEN_DISABLE_FLASH_SDPA", "1") == "1":
                    try:
                        torch.backends.cuda.enable_flash_sdp(False)
                        torch.backends.cuda.enable_mem_efficient_sdp(False)
                        print("  [INFO] Qwen SDPA backend: flash=False, mem_efficient=False, math=True")
                    except Exception as _sdpa_backend_err:
                        print(f"  [WARN] Failed to tune SDPA backend: {_sdpa_backend_err}")
            elif "llama" in _model_name_lower:
                _attn_impl = "eager"
            else:
                _attn_impl = "sdpa"
            print(f"  [INFO] attn_implementation={_attn_impl} (model: {os.path.basename(model_name)})")

            if torch.cuda.is_available():
                target_device = torch.device(device if device else "cuda:0")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=False,
                        attn_implementation=_attn_impl,
                    )
                    self.model.to(target_device)
                except Exception as e:
                    print(f"⚠️ 模型加载警告: {e}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=False,
                        attn_implementation=_attn_impl,
                    )
                    self.model.to(target_device)
                
                self.model.eval()
                print(f"✓ Model loaded successfully in eval mode on {target_device}")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
                self.model.eval()
        elif provider == 'openai':
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.client = OpenAI(
                api_key="sk-9geOE2ESZbKuWtgZt0F-Sg",
                base_url="https://llmapi.paratera.com/v1/"
            )

        elif provider == "vllm":
            import importlib
            vllm_mod = importlib.import_module(
                "vllm") if importlib.util.find_spec("vllm") else None
            if vllm_mod is None:
                raise ImportError(
                    "vllm is not installed. Please install vllm or use provider='huggingface'/'openai'.")
            LLM = getattr(vllm_mod, "LLM")
            
            current_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
            print(f"  [VLLM] 初始化模型，CUDA_VISIBLE_DEVICES={current_cuda_devices}...")
            
            self.model = LLM(
                model_name, 
                gpu_memory_utilization=0.90,
                tensor_parallel_size=1,
                trust_remote_code=True,
                max_model_len=16384,
                enforce_eager=True,
            )
            self.tokenizer = self.model.get_tokenizer()

    def query(self, prompt: str, **kwargs) -> Union[str, list]:
        if self.provider == 'openai':
            return self.query_openai(prompt, **kwargs)
        elif self.provider == 'huggingface':
            return self.query_huggingface(prompt, **kwargs)
        elif self.provider == "vllm":
            return self.query_vllm(prompt, **kwargs)
        else:
            raise ValueError("Unsupported provider")

    def query_with_timeout(self, messages, timeout=60, **kwargs):
        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def api_call():
            try:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **kwargs
                )
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)

        thread = threading.Thread(target=api_call)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise StopIteration("API call timed out")

        if not exception_queue.empty():
            raise exception_queue.get()

        if not result_queue.empty():
            return result_queue.get()

        raise StopIteration("API call failed unexpectedly")

    def query_openai(self,
                     prompt: str,
                     system: Optional[str] = None,
                     rate_limit_per_minute: Optional[int] = None, **kwargs) -> Union[str, list]:
        if "enable_thinking" in kwargs:
            kwargs.pop("enable_thinking", None)
        if system is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": prompt}]

        for i in range(64):
            try:
                kwargs['n'] = 1
                response = self.query_with_timeout(messages, **kwargs)

                if rate_limit_per_minute:
                    time.sleep(60 / rate_limit_per_minute - 0.5)

                if isinstance(response, str):
                    return response, response
                if kwargs.get('n', 1) == 1:
                    return response.choices[0].message.content, response
                else:
                    return [choice.message.content for choice in response.choices], response
            except StopIteration:
                print("Query timed out, retrying...")
                continue
            except Exception as e:
                print(e)
                time.sleep(10)

        raise RuntimeError("Failed to query the OpenAI API after 64 retries.")

    def query_huggingface(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.pop("temperature", 0.8)
        top_p = kwargs.pop("top_p", 1.0)
        max_new_tokens = kwargs.pop(
            "max_tokens", kwargs.pop("max_new_tokens", 256))
        n = int(kwargs.pop("n", 1))
        system = kwargs.pop("system", None)
        enable_thinking = kwargs.pop("enable_thinking", False)
        
        if isinstance(enable_thinking, str):
            enable_thinking = enable_thinking.strip().lower() in ("1", "true", "yes", "on")

        kwargs.pop("stop", None)
        do_sample = (temperature is not None) and (float(temperature) > 0)

        if getattr(self, "_has_chat_template", False):
            messages = []
            if system is not None:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            import inspect
            apply_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            try:
                sig = inspect.signature(self.tokenizer.apply_chat_template)
                if "enable_thinking" in sig.parameters:
                    apply_kwargs["enable_thinking"] = bool(enable_thinking)
            except Exception:
                pass

            text_inputs = self.tokenizer.apply_chat_template(
                messages, **apply_kwargs)
        else:
            if system is not None:
                text_inputs = f"System: {system}\nUser: {prompt}\nAssistant:"
            else:
                text_inputs = f"User: {prompt}\nAssistant:"

        enc = self.tokenizer(
            text_inputs, return_tensors="pt", padding=True,
            truncation=True, max_length=8192
        )
        
        device = next(self.model.parameters()).device

        enc = {k: v.to(device) for k, v in enc.items()}

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0]

        generate_kwargs = {
            "max_new_tokens": int(max_new_tokens) if max_new_tokens else 256,
            "num_return_sequences": n,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
        }
        
        if do_sample:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = float(temperature) if temperature is not None else 1.0
            generate_kwargs["top_p"] = float(top_p)
        else:
            generate_kwargs["do_sample"] = False

        if not enable_thinking:
            try:
                bad_words_ids = []
                for s in ("<think>", "</think>"):
                    ids = self.tokenizer.encode(s, add_special_tokens=False)
                    if ids:
                        bad_words_ids.append(ids)
                if bad_words_ids:
                    existing = generate_kwargs.get("bad_words_ids")
                    generate_kwargs["bad_words_ids"] = (
                        existing or []) + bad_words_ids
            except Exception:
                pass

        generate_kwargs.update(kwargs)

        with torch.no_grad(), torch.cuda.device(device):
            try:
                outputs = self.model.generate(**enc, **generate_kwargs)
            except RuntimeError as e:
                if "CUDA" in str(e) or "illegal memory access" in str(e) or "out of memory" in str(e).lower():
                    print(f"⚠️ CUDA error during generation: {e}")
                    print(f"⚠️ Input shape: {enc['input_ids'].shape}, max_new_tokens: {max_new_tokens}")
                    
                    try:
                        with torch.cuda.device(device):
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    
                    print("⚠️ Retrying with conservative parameters...")
                    if enc['input_ids'].shape[-1] > 4096:
                        enc = {k: v[:, -4096:] for k, v in enc.items()}
                        print(f"⚠️ Input truncated to 4096 tokens for retry")
                    simplified_kwargs = {
                        "max_new_tokens": min(128, int(max_new_tokens) if max_new_tokens else 128),
                        "do_sample": False,
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": pad_token_id,
                        "use_cache": True,
                    }
                    try:
                        outputs = self.model.generate(**enc, **simplified_kwargs)
                        print("✓ Retry successful")
                    except Exception as e2:
                        print(f"✗ Retry failed: {e2}")
                        raise RuntimeError(f"Generation failed after retry: {e2}") from e
                else:
                    raise

        prompt_len = enc["input_ids"].shape[-1]
        if outputs.dim() == 2:
            gen_tokens = []
            for i in range(outputs.size(0)):
                seq = outputs[i]
                gen_tokens.append(seq[prompt_len:])
        else:
            gen_tokens = [outputs]

        decoded_results = []
        thinking_info = {}

        for t in gen_tokens:
            output_ids = t.tolist()
            full_content = self.tokenizer.decode(
                output_ids, skip_special_tokens=True)

            if enable_thinking:
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                    thinking_content = self.tokenizer.decode(
                        output_ids[:index], skip_special_tokens=True).strip("\n")
                    answer_content = self.tokenizer.decode(
                        output_ids[index:], skip_special_tokens=True).strip("\n")

                    thinking_info = {
                        "has_thinking": True,
                        "thinking_length": len(thinking_content),
                        "thinking_content": thinking_content,
                        "answer_content": answer_content
                    }
                    decoded_results.append(full_content)

                except ValueError:
                    thinking_info = {"has_thinking": False}
                    decoded_results.append(full_content)
            else:
                thinking_info = {"has_thinking": False}
                decoded_results.append(full_content)

        if n == 1:
            result = decoded_results[0]
            metadata = {
                "prompt": prompt,
                "prompt_length": prompt_len,
                "enable_thinking": enable_thinking,
                "thinking_info": thinking_info
            }
            return result, metadata
        else:
            metadata = {
                "prompt": prompt,
                "prompt_length": prompt_len,
                "enable_thinking": enable_thinking,
                "thinking_info": thinking_info
            }
            return decoded_results, metadata

    def query_vllm(self, prompt: str, **kwargs) -> str:
        import importlib
        vllm_mod = importlib.import_module(
            "vllm") if importlib.util.find_spec("vllm") else None
        if vllm_mod is None:
            raise ImportError(
                "vllm is not installed. Please install vllm or use provider='huggingface'/'openai'.")
        SamplingParams = getattr(vllm_mod, "SamplingParams")

        n = kwargs.get("n", 1)
        system = kwargs.get("system", None)
        enable_thinking = kwargs.get("enable_thinking", False)
        if isinstance(enable_thinking, str):
            enable_thinking = enable_thinking.strip().lower() in ("1", "true", "yes", "on")
        
        max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 2048))
        temperature = kwargs.get("temperature", 0.8)
        top_p = kwargs.get("top_p", 1.0)
        
        sampling_params = SamplingParams(
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            stop=kwargs.get("stop", []),
            top_p=float(top_p) if temperature > 0 else 1.0,
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        )

        has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and getattr(
            self.tokenizer, "chat_template", None)
        
        if has_chat_template:
            messages = []
            if system is not None:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            import inspect
            apply_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            try:
                sig = inspect.signature(self.tokenizer.apply_chat_template)
                if "enable_thinking" in sig.parameters:
                    apply_kwargs["enable_thinking"] = bool(enable_thinking)
            except Exception:
                pass
            
            formatted_prompt = self.tokenizer.apply_chat_template(messages, **apply_kwargs)
        else:
            if system is not None:
                formatted_prompt = f"System: {system}\nUser: {prompt}\nAssistant:"
            else:
                formatted_prompt = f"User: {prompt}\nAssistant:"
        
        prompts = [formatted_prompt] * n

        try:
            outputs = self.model.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False
            )

            outputs = [output.outputs[0].text for output in outputs]
        except ValueError as e:
            print(f"VLLM generate error: {e}")
            outputs = ["Sorry, I don't know the answer to that question."]

        if n == 1:
            return outputs[0], {"prompt": prompts[0], "enable_thinking": enable_thinking}
        else:
            return outputs, {"prompt": prompts[0], "enable_thinking": enable_thinking}