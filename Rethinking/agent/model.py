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
                model_name, trust_remote_code=True, use_fast=False)
            # 记录是否存在聊天模板
            self._has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and getattr(
                self.tokenizer, "chat_template", None)
            # 处理缺省的 pad_token
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            # 注意：思考模式不在此处设置默认值，完全由调用者通过参数控制

            # 优先使用 bfloat16（Qwen3 推荐），不支持 bf16 时回退到 fp16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # 清理GPU缓存，避免碎片化
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 对于 Llama 3.1，强制使用 sdpa attention（避免CUDA illegal memory access）
            # eager 模式在 Llama 3.1 + RoPE 时会触发 CUDA 错误
            # flash_attention_2 需要额外安装，sdpa 是最稳定的选择
            if torch.cuda.is_available():
                try:
                    # 如果指定了device(如"cuda:0"或"cuda:1"),使用指定设备
                    # 否则使用device_map="auto"自动分配
                    if device:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            torch_dtype=dtype,
                            low_cpu_mem_usage=True,
                            attn_implementation="eager",  # 强制使用eager避免CUDA错误
                        ).to(device)
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            device_map="auto",
                            torch_dtype=dtype,
                            low_cpu_mem_usage=True,
                            max_memory={0: "76GB"},  # 为A800预留4GB显存
                            attn_implementation="eager",  # 强制使用eager避免CUDA错误
                        )
                except Exception as e:
                    print(f"⚠️ 模型加载警告: {e}")
                    # 回退到单卡加载，同样使用eager
                    target_device = device if device else "cuda"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        attn_implementation="eager",
                    ).to(target_device)
                
                # 设置为评估模式（关闭dropout等）
                self.model.eval()
                print(f"✓ Model loaded successfully in eval mode")
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
                # 师姐api_key="sk-pnL7eGjeAR39jSECFohDoQ",
                api_key="sk-9geOE2ESZbKuWtgZt0F-Sg",
                base_url="https://llmapi.paratera.com/v1/"
            )
            # self.client = OpenAI(魔塔
            #     api_key="ms-f5aaaf63-0596-473f-aa94-cfbe0468f7df",
            #     base_url="https://api-inference.modelscope.cn/v1/"
            # )
            # self.client = OpenAI(自己充
            #     api_key="sk-c34617fd48a54a6b8b6744f11896c4d9",
            #     base_url="https://api.deepseek.com"
            # )

            # #API_KEY = os.getenv("OPENAI_API_KEY", None)

            # if API_KEY is None:
            #     raise ValueError("OPENAI_API_KEY not set, please run `export OPENAI_API_KEY=<your key>` to ser it")
            # else:
            #     openai.api_key = API_KEY

        elif provider == "vllm":
            import importlib
            vllm_mod = importlib.import_module(
                "vllm") if importlib.util.find_spec("vllm") else None
            if vllm_mod is None:
                raise ImportError(
                    "vllm is not installed. Please install vllm or use provider='huggingface'/'openai'.")
            LLM = getattr(vllm_mod, "LLM")
            
            # 获取当前 CUDA_VISIBLE_DEVICES 设置（由调用者控制）
            current_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
            print(f"  [VLLM] 初始化模型，CUDA_VISIBLE_DEVICES={current_cuda_devices}...")
            
            # VLLM 会自动使用 CUDA_VISIBLE_DEVICES 中可见的 GPU
            # 关键参数：
            # - max_model_len: 限制最大序列长度，避免 KV 缓存不足
            # - gpu_memory_utilization: 显存利用率
            # - enforce_eager: 禁用 CUDA graph，节省显存
            self.model = LLM(
                model_name, 
                gpu_memory_utilization=0.90,  # 提高显存利用率
                tensor_parallel_size=1,  # 单卡运行
                trust_remote_code=True,
                max_model_len=16384,  # 限制最大序列长度，16K tokens 足够推理任务
                enforce_eager=True,  # 禁用 CUDA graph，节省显存
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
        """
        Windows-compatible timeout wrapper for API calls
        """
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
            # Thread is still running, timeout occurred
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
        # OpenAI Chat Completions 不支持自定义参数 enable_thinking，避免透传
        if "enable_thinking" in kwargs:
            kwargs.pop("enable_thinking", None)
        # Set default system message
        if system is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": prompt}]

        for i in range(64):
            try:
                # DeepSeek 只支持 n=1
                kwargs['n'] = 1
                response = self.query_with_timeout(messages, **kwargs)

                # Sleep to avoid rate limit if rate limit is set
                if rate_limit_per_minute:
                    # Buffer of 0.5 seconds
                    time.sleep(60 / rate_limit_per_minute - 0.5)

                # 兼容第三方API返回字符串的情况
                if isinstance(response, str):
                    return response, response
                if kwargs.get('n', 1) == 1:
                    return response.choices[0].message.content, response
                else:
                    return [choice.message.content for choice in response.choices], response
            except StopIteration:
                print("Query timed out, retrying...")
                continue  # Retry
            except Exception as e:
                print(e)
                time.sleep(10)

        raise RuntimeError("Failed to query the OpenAI API after 64 retries.")

    def query_huggingface(self, prompt: str, **kwargs) -> str:
        # 兼容 OpenAI 风格参数并转换为 HF generate 所需参数
        temperature = kwargs.pop("temperature", 0.8)
        top_p = kwargs.pop("top_p", 1.0)
        max_new_tokens = kwargs.pop(
            "max_tokens", kwargs.pop("max_new_tokens", 256))
        n = int(kwargs.pop("n", 1))
        # 读取 system，用于 chat 模板
        system = kwargs.pop("system", None)
        # Qwen3 思考模式开关（HF tokenizer.apply_chat_template >= 4.51.0 支持 enable_thinking，默认 True）
        # 这里优先使用调用者传入，其次默认 False（与脚本保持一致）。
        enable_thinking = kwargs.pop("enable_thinking", False)
        if isinstance(enable_thinking, str):
            enable_thinking = enable_thinking.strip().lower() in ("1", "true", "yes", "on")

        # 丢弃不适用于 HF generate 的参数
        kwargs.pop("stop", None)

        do_sample = (temperature is not None) and (float(temperature) > 0)

        # 构造输入（优先使用 chat 模板），并放到与模型相同的设备
        if getattr(self, "_has_chat_template", False):
            messages = []
            if system is not None:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # 准备 chat 模板参数（仅在支持时传入 enable_thinking）
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
                # 安全回退：不传该参数
                pass

            text_inputs = self.tokenizer.apply_chat_template(
                messages, **apply_kwargs)
        else:
            # 无模板时的通用回退（尽量贴近对话格式）
            if system is not None:
                text_inputs = f"System: {system}\nUser: {prompt}\nAssistant:"
            else:
                text_inputs = f"User: {prompt}\nAssistant:"

        enc = self.tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True)
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        # 确保 pad_token_id 是 int
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
        
        # 只有在 do_sample=True 时才添加 temperature 和 top_p
        if do_sample:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = float(temperature) if temperature is not None else 1.0
            generate_kwargs["top_p"] = float(top_p)
        else:
            generate_kwargs["do_sample"] = False
        # 兜底：若关闭思考但 tokenizer 不支持 enable_thinking 参数，则通过 bad_words_ids 屏蔽显式 <think> 标签
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
        # 合并其余兼容参数（如 repetition_penalty 等）
        generate_kwargs.update(kwargs)

        # 清理CUDA缓存，避免内存碎片
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(**enc, **generate_kwargs)
            except RuntimeError as e:
                # 如果出现CUDA错误，清理缓存并重试
                if "CUDA" in str(e) or "illegal memory access" in str(e):
                    print(f"⚠️ CUDA error during generation: {e}")
                    print(f"⚠️ Input shape: {enc['input_ids'].shape}, max_new_tokens: {max_new_tokens}")
                    
                    # 强制同步和清理
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    # 使用更保守的生成参数重试
                    print("⚠️ Retrying with conservative parameters...")
                    simplified_kwargs = {
                        "max_new_tokens": min(64, int(max_new_tokens) if max_new_tokens else 64),
                        "do_sample": False,  # 禁用采样，使用greedy
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": pad_token_id,
                        "use_cache": True,  # 启用KV缓存
                    }
                    try:
                        outputs = self.model.generate(**enc, **simplified_kwargs)
                        print("✓ Retry successful")
                    except Exception as e2:
                        print(f"✗ Retry failed: {e2}")
                        raise RuntimeError(f"Generation failed after retry: {e2}") from e
                else:
                    raise

        # outputs 形状：[n, seq_len]，包含 prompt + 新 tokens，需要切掉 prompt 部分
        prompt_len = enc["input_ids"].shape[-1]
        # 对于部分模型，generate 可能返回不同长度，逐个切片更稳妥
        if outputs.dim() == 2:
            gen_tokens = []
            for i in range(outputs.size(0)):
                seq = outputs[i]
                gen_tokens.append(seq[prompt_len:])
        else:
            # 退化情况，直接整体解码
            gen_tokens = [outputs]

        # 根据官方文档解析思考内容（仅当启用思考模式时）
        decoded_results = []
        thinking_info = {}

        for t in gen_tokens:
            output_ids = t.tolist()
            full_content = self.tokenizer.decode(
                output_ids, skip_special_tokens=True)

            if enable_thinking:
                # 尝试解析思考内容，参考官方文档示例
                try:
                    # 查找 </think> 标记 (token id: 151668)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                    thinking_content = self.tokenizer.decode(
                        output_ids[:index], skip_special_tokens=True).strip("\n")
                    answer_content = self.tokenizer.decode(
                        output_ids[index:], skip_special_tokens=True).strip("\n")

                    # 记录思考信息
                    thinking_info = {
                        "has_thinking": True,
                        "thinking_length": len(thinking_content),
                        "thinking_content": thinking_content,
                        "answer_content": answer_content
                    }

                    # 返回完整内容，包含思考过程
                    decoded_results.append(full_content)

                except ValueError:
                    # 没有找到 </think> 标记
                    thinking_info = {"has_thinking": False}
                    decoded_results.append(full_content)
            else:
                # 非思考模式，直接解码
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
        
        # 获取 max_tokens，支持多种参数名
        max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 2048))
        temperature = kwargs.get("temperature", 0.8)
        top_p = kwargs.get("top_p", 1.0)
        
        # 构造采样参数
        sampling_params = SamplingParams(
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            stop=kwargs.get("stop", []),
            top_p=float(top_p) if temperature > 0 else 1.0,
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        )

        # 检查 tokenizer 是否有 chat template
        has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and getattr(
            self.tokenizer, "chat_template", None)
        
        if has_chat_template:
            # 使用 Qwen3 的 chat template
            messages = []
            if system is not None:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            # 准备 chat 模板参数
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
            # 使用通用格式
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
