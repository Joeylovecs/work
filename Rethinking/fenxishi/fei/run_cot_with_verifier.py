"""
分析师辅助推理框架
使用微调后的Qwen3-8B作为分析师,辅助原始Qwen3-8B进行表格推理
"""
import os
import sys
import json
from typing import Optional, Tuple, List
from tqdm import tqdm
from fire import Fire

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from agent import Model
from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes
from utils.table import transpose, sort_dataframe
from run_helper import load_dataset, get_cot_prompt, query, check_transpose, check_sort, read_json_file


# 分析师prompt模板
VERIFIER_PROMPT_TEMPLATE = """You are a table reasoning expert, capable of analyzing and understanding information within tables. You are also skilled at verifying and analyzing reasoning steps. Your task is to critically evaluate a given set of reasoning steps that attempt to answer a question based on a provided table.

You must determine if the reasoning process is correct and logically sound.

**Provided Information:**

**Table:**
{table}

**Question:**
{question}

**Reasoning Steps to Analyze:**
{reasoning_steps}

---

**Your Task:**

**Phase 1: Verification**
Carefully verify each reasoning step against the table and the question. Check for errors in data interpretation, calculation, logical deduction, and final conclusion.

**Phase 2: Judgment**
Based on your verification, determine if the reasoning is CORRECT or INCORRECT.

**Output Format:**

If the reasoning is CORRECT, output:
VERIFICATION RESULT: CORRECT

If the reasoning contains ANY ERROR, output:
VERIFICATION RESULT: INCORRECT

[Error Analysis]
- **Step with Error**: Step [number]
- **Error Description**: [Describe what went wrong]
- **Suggestion**: [How to correct this step]

Please provide your judgment now.
"""


def truncate_base_model_output(text: str) -> str:
    """
    截断基座模型输出,去除重复的Final Answer
    
    问题: 基座模型经常输出 "Final Answer: xxx" 重复几百次
    解决: 找到第一个完整的Final Answer后立即截断
    """
    if not text:
        return text
    
    lines = text.strip().split('\n')
    result_lines = []
    found_final_answer = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # 如果这行包含Final Answer
        if 'Final Answer:' in line_stripped or 'Final Answer：' in line_stripped:
            if not found_final_answer:
                # 第一次遇到Final Answer,保留
                result_lines.append(line)
                found_final_answer = True
            # 后续的Final Answer全部丢弃
            continue
        
        # 检测重复模式: 如果这行和前一行完全相同,跳过
        if result_lines and line_stripped == result_lines[-1].strip():
            continue
            
        result_lines.append(line)
    
    return '\n'.join(result_lines)


def extract_reasoning_steps(text: str) -> str:
    """
    从模型输出中提取推理步骤
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # 首先截断重复输出
    text = truncate_base_model_output(text)
    
    # 尝试提取Step 1到Final Answer之间的内容
    lines = text.strip().split('\n')
    reasoning_lines = []
    in_reasoning = False
    
    for line in lines:
        line_stripped = line.strip()
        # 开始收集推理步骤
        if line_stripped.startswith('Step 1'):
            in_reasoning = True
        
        if in_reasoning:
            reasoning_lines.append(line)
            
            # 如果遇到Final Answer或Therefore,继续收集这一行然后停止
            if 'Final Answer:' in line_stripped or line_stripped.startswith('Therefore'):
                break
    
    return '\n'.join(reasoning_lines) if reasoning_lines else text


def verify_reasoning(verifier_model: Model, table: str, question: str, reasoning_steps: str, 
                    system: str = "You are a helpful assistant") -> Tuple[bool, str]:
    """
    使用分析师模型验证推理步骤
    
    Args:
        verifier_model: 分析师模型(微调后的Qwen)
        table: 表格内容
        question: 问题
        reasoning_steps: 推理步骤
        system: 系统提示
    
    Returns:
        (is_correct, feedback): 是否正确以及反馈信息
    """
    # 构造验证prompt
    verifier_prompt = VERIFIER_PROMPT_TEMPLATE.format(
        table=table,
        question=question,
        reasoning_steps=reasoning_steps
    )
    
    # 调用分析师模型
    # 分析师反馈使用max_tokens=2048,并添加repetition_penalty防止重复
    verification_text, _ = verifier_model.query(
        prompt=verifier_prompt,
        temperature=0.1,
        max_tokens=2048,  # 保持2048
        n=1,
        system=system,
        enable_thinking=False,
        repetition_penalty=1.2  # 防止重复生成
    )
    
    if verification_text is None:
        return True, "Verification failed, assuming correct"
    
    # 后处理:截断分析师输出,只保留有效内容
    # 分析师输出格式应该是:
    # VERIFICATION RESULT: CORRECT/INCORRECT
    # [Error Analysis] (如果INCORRECT)
    # - **Step with Error**: ...
    # - **Error Description**: ...
    # - **Suggestion**: ...
    verification_text = truncate_verifier_output(verification_text)
    
    # 解析验证结果
    is_correct = "VERIFICATION RESULT: CORRECT" in verification_text
    
    return is_correct, verification_text


def truncate_verifier_output(text: str) -> str:
    """
    截断分析师输出,只保留有效的验证内容
    
    分析师输出应该在 **Suggestion**: ... 之后结束
    如果检测到重复模式(如多个"Final Answer"),则截断
    """
    if not text:
        return text
    
    lines = text.strip().split('\n')
    result_lines = []
    found_suggestion = False
    suggestion_line_count = 0
    
    for line in lines:
        line_stripped = line.strip()
        
        # 检测到Suggestion行
        if '**Suggestion**' in line or line_stripped.startswith('- **Suggestion'):
            found_suggestion = True
        
        # 如果已经找到Suggestion,只再收集最多9行(允许Suggestion内容跨行)
        if found_suggestion:
            suggestion_line_count += 1
            result_lines.append(line)
            if suggestion_line_count > 9:
                break
        else:
            # 检测重复模式:如果遇到Final Answer说明模型开始重复,截断
            if line_stripped.startswith('**Final Answer'):
                break
            result_lines.append(line)
    
    return '\n'.join(result_lines)


def reasoning_with_verifier(
    base_model: Model,
    long_model: Model,
    verifier_model: Model,
    prompt: str,
    table: str,
    question: str,
    temperature: float,
    self_consistency: int,
    max_iterations: int = 3,  # 默认改为3次
    system: str = "You are a helpful assistant",
    enable_thinking: bool = False
) -> Tuple[str, List[dict]]:
    """
    使用分析师辅助的推理过程
    
    Args:
        base_model: 基础推理模型
        long_model: 长上下文模型(实际复用base_model)
        verifier_model: 分析师模型(微调后的Qwen)
        prompt: 推理prompt
        table: 表格内容
        question: 问题
        temperature: 温度参数
        self_consistency: 自一致性参数
        max_iterations: 最大重试次数
        system: 系统提示
        enable_thinking: 是否启用思考模式
    
    Returns:
        (final_text, iteration_history): 最终推理结果和迭代历史
    """
    iteration_history = []
    best_text = None  # 保存最佳结果
    
    for iteration in range(max_iterations):
        # 构造当前迭代的prompt
        if iteration == 0:
            # 第一次推理:使用原始prompt
            current_prompt = prompt
        else:
            # 后续推理:只使用上一次的反馈,避免prompt爆炸
            # 关键改进: 不再累积所有历史,只保留最近一次反馈
            last_hist = iteration_history[-1]
            
            # 提取精简的反馈信息
            feedback = last_hist['verification']
            # 进一步精简反馈,只保留关键部分
            feedback_lines = feedback.split('\n')
            short_feedback = []
            for line in feedback_lines[:15]:  # 最多15行反馈
                if line.strip():
                    short_feedback.append(line)
            feedback = '\n'.join(short_feedback)
            
            history_context = f"""

=== CORRECTION NEEDED ===
Your previous answer was marked INCORRECT by the analyst.

Previous Answer: {extract_final_answer_from_text(last_hist.get('reasoning', ''))}

Analyst Feedback:
{feedback}

INSTRUCTIONS:
1. Read the feedback carefully
2. Fix the specific error mentioned
3. Provide corrected reasoning

Now give your corrected answer:
"""
            current_prompt = prompt + history_context
        
        # 1. 基座模型推理
        text, response = query(
            base_model, long_model, current_prompt, 
            temperature, self_consistency, 
            system=system, enable_thinking=enable_thinking
        )
        
        if text is None:
            # 推理失败,记录并返回
            iteration_history.append({
                "iteration": iteration + 1,
                "reasoning": None,
                "verification": "Reasoning failed",
                "is_correct": False
            })
            # 如果有之前的结果,返回之前的
            return best_text if best_text else text, iteration_history
        
        # 截断重复输出
        text = truncate_base_model_output(text)
        
        # 清理异常前缀(如.ipv, ió, .onerror等)
        text = text.strip()
        lines = text.split('\n')
        if lines and len(lines[0]) < 20 and not lines[0].startswith('Step'):
            # 第一行太短且不是Step开头,可能是异常前缀,删除
            text = '\n'.join(lines[1:]).strip()
        
        # 保存结果
        if best_text is None:
            best_text = text
        
        # 提取推理步骤
        reasoning_steps = extract_reasoning_steps(text)
        
        # 2. 分析师验证
        is_correct, verification_feedback = verify_reasoning(
            verifier_model, table, question, reasoning_steps, system
        )
        
        # 记录本次迭代
        iteration_history.append({
            "iteration": iteration + 1,
            "reasoning": text,
            "reasoning_steps": reasoning_steps,
            "verification": verification_feedback,
            "is_correct": is_correct
        })
        
        if is_correct:
            # 验证通过,返回结果
            return text, iteration_history
        else:
            # 更新最佳结果为最新的
            best_text = text
    
    # 超过最大迭代次数,返回最后一次的结果
    return text, iteration_history


def extract_final_answer_from_text(text: str) -> str:
    """从推理文本中提取最终答案"""
    if not text:
        return "N/A"
    
    # 查找 Final Answer: xxx
    import re
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text)
    if match:
        return match.group(1).strip()
    
    # 查找 Therefore, the answer is xxx
    match = re.search(r'Therefore.*?(?:is|:)\s*(.+?)(?:\n|$)', text)
    if match:
        return match.group(1).strip()
    
    return "N/A"


def main(
    # 基础推理模型
    model: Optional[str] = "Qwen3-8B",
    # 长上下文模型
    long_model: Optional[str] = "Qwen3-8B",
    # 分析师模型路径(微调后的模型)
    verifier_model_path: str = "buzhou/training_data_zhengti/weitiao/qwen3_8b_merged_final",
    provider: str = "huggingface",  # openai, huggingface, vllm
    dataset: str = "wtq",  # wtq or tabfact or train
    perturbation: str = "none",  # none, transpose, shuffle, transpose_shuffle
    norm: bool = True,  # whether to NORM the table
    disable_resort: bool = True,  # whether to disable the resort stage in NORM
    norm_cache: bool = True,  # whether to cache the normalization results
    sub_sample: bool = False,  # whether to only run on the subset sampled data points
    resume: int = 0,  # resume from the i-th data point
    stop_at: int = 1e6,  # stop at the i-th data point
    self_consistency: int = 1,  # how many times to do self consistency
    temperature: float = 0.1,  # temperature for model
    max_iterations: int = 3,  # 最大重试次数,默认3次
    log_dir: str = "fenxishi/output/wtq_with_verifier",  # directory to store the logs
    cache_dir: str = "cache",  # directory to store the cache
    system: str = "You are a helpful assistant",  # system prompt
    use_strict_format: bool = False,  # whether to use strict format prompt
    enable_thinking: bool = False,  # whether to enable thinking mode
):
    """
    使用分析师辅助推理的主函数
    """
    # 确保 enable_thinking 是布尔值
    if isinstance(enable_thinking, str):
        enable_thinking = enable_thinking.lower() in ("true", "1", "yes", "on")
    enable_thinking = bool(enable_thinking)

    print("=" * 80)
    print("分析师辅助推理框架")
    print("=" * 80)
    print(f"基础模型: {model}")
    print(f"长上下文模型: {long_model}")
    print(f"分析师模型: {verifier_model_path}")
    print(f"数据集: {dataset}")
    print(f"最大重试次数: {max_iterations}")
    print(f"输出目录: {log_dir}")
    print("=" * 80)

    #### create log & cache dir and save config ####
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # store the config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w", encoding='utf-8') as f:
        json.dump({key: value for key, value in locals().items()
                  if key != 'f'}, f, indent=4, ensure_ascii=False)

    #### load dataset and cot prompt ####
    data = load_dataset(dataset)
    cot_prompt = get_cot_prompt(dataset, use_strict_format)

    #### load the models ####
    print("\n加载模型...")
    
    if model:
        print(f"  加载基础模型到GPU 0: {model}")
        model = Model(model, provider=provider, device="cuda:0")
        # 基座模型同时用作长上下文模型,不需要重复加载
        long_model = model
        print(f"    -> 基座模型已加载到GPU 0,同时处理长上下文(无需重复加载)")
    
    # 加载分析师模型到GPU 1
    print(f"  加载分析师模型到GPU 1: {verifier_model_path}")
    verifier_model = Model(verifier_model_path, provider="huggingface", device="cuda:1")
    
    print("模型加载完成!\n")
    print("GPU分配: 基座模型(GPU 0) + 分析师模型(GPU 1)\n")

    #### load the cache ####
    transpose_cache = read_json_file(os.path.join(cache_dir, "transpose.json"))
    resort_cache = read_json_file(os.path.join(cache_dir, "resort.json"))

    #### prepare the iterator ####
    global_i = 0
    break_flag = False
    total = sum([len(d['sampled_indices']) for d in data]) if sub_sample else sum(
        [len(d['questions']) for d in data])
    pbar = tqdm(total=stop_at if stop_at < total else total, desc="推理进度")

    # 统计信息
    stats = {
        "total": 0,
        "first_try_correct": 0,
        "corrected_after_retry": 0,
        "failed_after_max_retries": 0,
        "total_iterations": 0
    }

    #### start the loop ####
    for table_idx, d in enumerate(data):
        if break_flag:
            break

        index_list = d['sampled_indices'] if sub_sample else range(
            len(d["questions"]))

        if len(index_list) == 0:
            continue

        # load table infos
        table_id = d["table_id"]
        title = d["title"]

        if perturbation == "none":
            table = construct_markdown_table(**d["table"])
        elif perturbation == "transpose":
            table = construct_markdown_table(**d["transposed_table"])
        elif perturbation == "shuffle":
            table = construct_markdown_table(**d["row_shuffled_table"])
        elif perturbation == "transpose_shuffle":
            table = construct_markdown_table(**d["row_shuffled_transposed_table"])

        df = markdown_to_df(table)

        # transpose and sort if necessary
        transpose_flag = False
        resort_list = []

        if norm:
            transpose_flag = check_transpose(
                model, long_model, table, title, table_id, perturbation, 
                transpose_cache, norm_cache, cache_dir)

            if transpose_flag:
                transposed_df = transpose(df)
                df = remove_merged_suffixes(transposed_df)

            if not disable_resort:
                resort_list = check_sort(
                    model, long_model, df, title, table_id, perturbation, 
                    resort_cache, norm_cache, cache_dir)
                df = sort_dataframe(df, resort_list)

        # reset the table
        table = df.to_markdown()

        for idx in index_list:
            if global_i < resume:
                global_i += 1
                pbar.update(1)
                continue
            elif global_i >= stop_at:
                break_flag = True
                break

            question = d["questions"][idx]
            answer = d["answers"][idx]
            question_id = d["ids"][idx]

            prompt = cot_prompt.replace("[TABLE]", table)\
                .replace("[QUESTION]", question)\
                .replace("[TITLE]", title)\
                .strip()

            # 使用分析师辅助推理
            text, iteration_history = reasoning_with_verifier(
                base_model=model,
                long_model=long_model,
                verifier_model=verifier_model,
                prompt=prompt,
                table=table,
                question=question,
                temperature=temperature,
                self_consistency=self_consistency,
                max_iterations=max_iterations,
                system=system,
                enable_thinking=enable_thinking
            )

            # 更新统计信息
            stats["total"] += 1
            stats["total_iterations"] += len(iteration_history)
            
            if len(iteration_history) > 0:
                if iteration_history[0]["is_correct"]:
                    stats["first_try_correct"] += 1
                elif iteration_history[-1]["is_correct"]:
                    stats["corrected_after_retry"] += 1
                else:
                    stats["failed_after_max_retries"] += 1

            # 保存详细日志
            log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding='utf-8') as f:
                f.write("===================Title===================\n")
                f.write(title + "\n")
                f.write("===================Table===================\n")
                f.write(table + "\n")
                f.write("===================Question===================\n")
                f.write(question + "\n")
                f.write("===================Answer===================\n")
                f.write(",".join(answer) if isinstance(answer, list) else str(answer))
                f.write("\n\n")
                
                # 记录迭代历史
                f.write("===================Iteration History===================\n")
                f.write(f"说明: 每次迭代,基座模型会看到之前所有失败尝试的完整历史\n\n")
                
                for iter_info in iteration_history:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"迭代轮次: {iter_info['iteration']}/{max_iterations}\n")
                    if iter_info['iteration'] > 1:
                        f.write(f"(本次推理已累积前 {iter_info['iteration']-1} 次失败历史作为上下文)\n")
                    f.write(f"{'='*60}\n\n")
                    
                    # 基座模型推理
                    f.write(f"【基座模型推理输出】\n")
                    f.write(f"{'-'*60}\n")
                    f.write(f"{iter_info.get('reasoning', 'N/A')}\n")
                    f.write(f"{'-'*60}\n\n")
                    
                    # 分析师模型验证
                    f.write(f"【分析师模型验证反馈】\n")
                    f.write(f"{'-'*60}\n")
                    f.write(f"验证结果: {'✓ CORRECT' if iter_info['is_correct'] else '✗ INCORRECT'}\n\n")
                    f.write(f"详细分析:\n{iter_info.get('verification', 'N/A')}\n")
                    f.write(f"{'-'*60}\n")
                    
                    if iter_info['is_correct']:
                        f.write(f"\n✓ 验证通过,采用此次推理结果\n")
                    elif iter_info['iteration'] == max_iterations:
                        f.write(f"\n⚠ 已达最大迭代次数({max_iterations}次),保留第{max_iterations}次结果(即使标记为INCORRECT)\n")
                    else:
                        f.write(f"\n↻ 将此次错误反馈累积到历史,开始第{iter_info['iteration']+1}次推理...\n")
                
                f.write("\n===================Final Text===================\n")
                f.write(str(text))
                f.write("\n")

            # 保存结果到jsonl
            res = {
                "idx": global_i,
                "answer": answer,
                "text": text,
                "transpose": transpose_flag,
                "resort": resort_list,
                "question_id": question_id,
                "table_id": table_id,
                "title": title,
                "table": table,
                "question": question,
                "iteration_count": len(iteration_history),
                "first_try_correct": iteration_history[0]["is_correct"] if iteration_history else False,
                "final_correct": iteration_history[-1]["is_correct"] if iteration_history else False,
            }

            with open(os.path.join(log_dir, "result.jsonl"), "a", encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False)
                f.write("\n")

            global_i += 1
            pbar.update(1)
            
            # 更新进度条描述显示统计信息
            if stats["total"] > 0:
                first_acc = stats["first_try_correct"] / stats["total"] * 100
                pbar.set_postfix({
                    "首次正确率": f"{first_acc:.1f}%",
                    "重试成功": stats["corrected_after_retry"],
                    "平均迭代": f"{stats['total_iterations']/stats['total']:.1f}"
                })

    pbar.close()

    # 保存统计信息
    stats_path = os.path.join(log_dir, "statistics.json")
    with open(stats_path, "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("推理完成! 统计信息:")
    print("=" * 80)
    print(f"总问题数: {stats['total']}")
    print(f"首次正确: {stats['first_try_correct']} ({stats['first_try_correct']/stats['total']*100:.2f}%)")
    print(f"重试后正确: {stats['corrected_after_retry']} ({stats['corrected_after_retry']/stats['total']*100:.2f}%)")
    print(f"最终失败: {stats['failed_after_max_retries']} ({stats['failed_after_max_retries']/stats['total']*100:.2f}%)")
    print(f"平均迭代次数: {stats['total_iterations']/stats['total']:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    Fire(main)
