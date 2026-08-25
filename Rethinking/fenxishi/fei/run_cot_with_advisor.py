"""
分析师建议增强模式（Advisor Mode）
核心改进:
1. 分析师只提供建议，不做正误判断
2. 基座模型被强制重新审视，但可以坚持原答案
3. 限制prompt长度，只保留精简反馈
4. 对长表格进行智能压缩
"""
import os
import sys
import json
import re
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


# ============================================================
# 分析师Prompt: 只提供建议，不做判断
# ============================================================
ADVISOR_PROMPT_TEMPLATE = """You are a table reasoning advisor. Your job is to provide helpful suggestions to improve the answer, NOT to judge if the answer is correct or incorrect.

**Table:**
{table}

**Question:**
{question}

**Current Reasoning:**
{reasoning_steps}

---

**Your Task:** Provide 1-2 brief suggestions that might help improve the answer. Focus on:
- Any potential calculation errors
- Any data that might have been overlooked
- Any alternative interpretations

**IMPORTANT RULES:**
1. Do NOT say "CORRECT" or "INCORRECT"
2. Do NOT repeat the full table or question
3. Keep your response under 100 words
4. Be specific and actionable

**Your Suggestions:**
"""


# ============================================================
# 表格处理函数（保持完整，不压缩）
# ============================================================
def get_table_info(table: str) -> dict:
    """
    获取表格基本信息（用于日志记录）
    
    Args:
        table: markdown表格
    
    Returns:
        表格信息字典
    """
    lines = table.strip().split('\n')
    data_lines = [l for l in lines if l.strip() and '|' in l and not l.strip().startswith('|-')]
    
    num_rows = max(0, len(data_lines) - 1)  # 减去header行
    num_cols = 0
    if data_lines:
        num_cols = len([c for c in data_lines[0].split('|') if c.strip()])
    
    return {
        "rows": num_rows,
        "cols": num_cols,
        "char_length": len(table)
    }


# ============================================================
# 输出清理函数
# ============================================================
def truncate_output(text: str, max_final_answers: int = 1) -> str:
    """
    截断输出，去除重复的Final Answer
    """
    if not text:
        return text
    
    lines = text.strip().split('\n')
    result_lines = []
    final_answer_count = 0
    
    for line in lines:
        line_stripped = line.strip()
        
        # 检测Final Answer
        if 'Final Answer:' in line_stripped or 'Final Answer：' in line_stripped:
            final_answer_count += 1
            if final_answer_count <= max_final_answers:
                result_lines.append(line)
            continue
        
        # 跳过连续重复行
        if result_lines and line_stripped == result_lines[-1].strip():
            continue
        
        # 跳过异常前缀行
        if len(line_stripped) < 10 and not any(line_stripped.startswith(prefix) for prefix in ['Step', 'Therefore', 'The', 'I ']):
            if line_stripped in ['.ipv', '.ov', 'ió', '.onerror', 'RecognitionException', '.IGNORED']:
                continue
        
        result_lines.append(line)
    
    return '\n'.join(result_lines)


def extract_final_answer(text: str) -> str:
    """从推理文本中提取最终答案"""
    if not text:
        return "N/A"
    
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'Therefore.*?(?:is|:)\s*(.+?)(?:\n|$)', text)
    if match:
        return match.group(1).strip()
    
    return "N/A"


def extract_reasoning_steps(text: str) -> str:
    """提取推理步骤（精简版）"""
    if not text:
        return ""
    
    text = truncate_output(text)
    
    # 只保留 Step 开头的行和 Final Answer
    lines = text.strip().split('\n')
    key_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('Step') or 'Final Answer' in line_stripped or line_stripped.startswith('Therefore'):
            key_lines.append(line)
    
    # 如果没有找到Step格式，返回前10行
    if not key_lines:
        return '\n'.join(lines[:10])
    
    return '\n'.join(key_lines)


# ============================================================
# 分析师建议获取
# ============================================================
def get_advisor_suggestions(
    advisor_model: Model, 
    table: str, 
    question: str, 
    reasoning_steps: str,
    system: str = "You are a helpful assistant"
) -> str:
    """
    获取分析师建议（不做正误判断）
    """
    # 精简推理步骤（保持表格完整）
    short_reasoning = extract_reasoning_steps(reasoning_steps)
    if len(short_reasoning) > 1000:
        short_reasoning = short_reasoning[:1000] + "\n... (truncated)"
    
    # 构造prompt（使用完整表格）
    advisor_prompt = ADVISOR_PROMPT_TEMPLATE.format(
        table=table,
        question=question,
        reasoning_steps=short_reasoning
    )
    
    # 调用分析师模型
    suggestion_text, _ = advisor_model.query(
        prompt=advisor_prompt,
        temperature=0.1,
        max_tokens=256,  # 限制输出长度
        n=1,
        system=system,
        enable_thinking=False,
        repetition_penalty=1.3  # 更强的重复惩罚
    )
    
    if suggestion_text is None:
        return "Consider double-checking your calculation."
    
    # 清理和截断建议
    suggestion_text = truncate_output(suggestion_text)
    if len(suggestion_text) > 300:
        suggestion_text = suggestion_text[:300] + "..."
    
    return suggestion_text


# ============================================================
# 核心推理函数：建议增强模式
# ============================================================
def reasoning_with_advisor(
    base_model: Model,
    long_model: Model,
    advisor_model: Model,
    prompt: str,
    table: str,
    question: str,
    temperature: float,
    self_consistency: int,
    max_iterations: int = 2,  # 默认2轮（首次 + 1次建议增强）
    system: str = "You are a helpful assistant",
    enable_thinking: bool = False
) -> Tuple[str, List[dict], bool]:
    """
    建议增强模式推理
    
    核心设计：
    1. 第一轮：正常推理
    2. 后续轮：展示建议，要求重新审视（但允许坚持原答案）
    
    Returns:
        (final_text, iteration_history, kept_first_answer)
    """
    iteration_history = []
    first_answer = None
    first_text = None
    
    for iteration in range(max_iterations):
        
        # 构造当前迭代的prompt
        if iteration == 0:
            # 第一轮：正常推理
            current_prompt = prompt
        else:
            # 后续轮：建议增强模式
            # 关键设计：不说"你的答案错了"，而是"请再检查一下"
            last_suggestion = iteration_history[-1].get('suggestion', '')
            
            # 构造增强prompt - 允许坚持原答案
            enhancement = f"""

=== REVIEW REQUEST ===
Please review your answer. An advisor has provided the following suggestions:

{last_suggestion}

INSTRUCTIONS:
1. Consider the suggestions carefully
2. Re-examine the relevant parts of the table
3. You may KEEP your original answer if you believe it's correct
4. Or REVISE your answer if you find an error

Your original answer was: {first_answer}

Please provide your final answer (you may keep or change it):
"""
            current_prompt = prompt + enhancement
        
        # 检查prompt长度
        if len(current_prompt) > 25000:
            # Prompt太长，直接返回第一轮结果
            if first_text:
                return first_text, iteration_history, True
            else:
                return None, iteration_history, False
        
        # 基座模型推理
        text, response = query(
            base_model, long_model, current_prompt,
            temperature, self_consistency,
            system=system, enable_thinking=enable_thinking
        )
        
        if text is None:
            iteration_history.append({
                "iteration": iteration + 1,
                "reasoning": None,
                "error": "Query failed"
            })
            if first_text:
                return first_text, iteration_history, True
            return None, iteration_history, False
        
        # 清理输出
        text = truncate_output(text)
        
        # 提取答案
        current_answer = extract_final_answer(text)
        
        if iteration == 0:
            first_answer = current_answer
            first_text = text
        
        # 记录本次迭代
        iter_info = {
            "iteration": iteration + 1,
            "reasoning": text,
            "answer": current_answer
        }
        
        # 如果不是最后一轮，获取建议
        if iteration < max_iterations - 1:
            suggestion = get_advisor_suggestions(
                advisor_model, table, question, text, system
            )
            iter_info["suggestion"] = suggestion
        
        iteration_history.append(iter_info)
    
    # 检查最终答案是否与第一轮相同
    final_answer = extract_final_answer(text)
    kept_first = (final_answer == first_answer)
    
    return text, iteration_history, kept_first


# ============================================================
# 主函数
# ============================================================
def main(
    model: Optional[str] = "Qwen3-8B",
    long_model: Optional[str] = "Qwen3-8B",
    advisor_model_path: str = "buzhou/training_data_zhengti/weitiao/qwen3_8b_merged_final",
    provider: str = "huggingface",
    dataset: str = "wtq",
    perturbation: str = "none",
    norm: bool = True,
    disable_resort: bool = True,
    norm_cache: bool = True,
    sub_sample: bool = False,
    resume: int = 0,
    stop_at: int = 1e6,
    self_consistency: int = 1,
    temperature: float = 0.1,
    max_iterations: int = 2,  # 建议增强模式默认2轮
    log_dir: str = "fenxishi/output/wtq_advisor",
    cache_dir: str = "cache",
    system: str = "You are a helpful assistant",
    use_strict_format: bool = False,
    enable_thinking: bool = False,
):
    """
    分析师建议增强模式主函数
    """
    if isinstance(enable_thinking, str):
        enable_thinking = enable_thinking.lower() in ("true", "1", "yes", "on")
    enable_thinking = bool(enable_thinking)

    print("=" * 80)
    print("分析师建议增强模式 (Advisor Mode)")
    print("=" * 80)
    print(f"基础模型: {model}")
    print(f"分析师模型: {advisor_model_path}")
    print(f"数据集: {dataset}")
    print(f"增强轮次: {max_iterations}")
    print(f"输出目录: {log_dir}")
    print("=" * 80)
    print("\n模式说明:")
    print("  - 分析师只提供建议，不做正误判断")
    print("  - 基座模型可以坚持原答案或修改")
    print("  - Prompt长度受限，自动压缩表格")
    print("=" * 80 + "\n")

    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "log"), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # 保存配置
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w", encoding='utf-8') as f:
        json.dump({key: value for key, value in locals().items() if key != 'f'}, f, indent=4, ensure_ascii=False)

    # 加载数据集和prompt
    data = load_dataset(dataset)
    cot_prompt = get_cot_prompt(dataset, use_strict_format)

    # 加载模型
    print("\n加载模型...")
    print(f"  基座模型 (GPU 0): {model}")
    model = Model(model, provider=provider, device="cuda:0")
    long_model = model
    
    print(f"  分析师模型 (GPU 1): {advisor_model_path}")
    advisor_model = Model(advisor_model_path, provider="huggingface", device="cuda:1")
    print("模型加载完成!\n")

    # 加载缓存
    transpose_cache = read_json_file(os.path.join(cache_dir, "transpose.json"))
    resort_cache = read_json_file(os.path.join(cache_dir, "resort.json"))

    # 统计信息
    stats = {
        "total": 0,
        "kept_first_answer": 0,
        "changed_answer": 0,
        "prompt_too_long": 0,
        "total_iterations": 0
    }

    # 迭代处理
    global_i = 0
    break_flag = False
    total = sum([len(d['sampled_indices']) for d in data]) if sub_sample else sum([len(d['questions']) for d in data])
    pbar = tqdm(total=min(stop_at, total), desc="推理进度")

    for table_idx, d in enumerate(data):
        if break_flag:
            break

        index_list = d['sampled_indices'] if sub_sample else range(len(d["questions"]))
        if len(index_list) == 0:
            continue

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

        # Normalization
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

            # 使用建议增强模式推理
            text, iteration_history, kept_first = reasoning_with_advisor(
                base_model=model,
                long_model=long_model,
                advisor_model=advisor_model,
                prompt=prompt,
                table=table,
                question=question,
                temperature=temperature,
                self_consistency=self_consistency,
                max_iterations=max_iterations,
                system=system,
                enable_thinking=enable_thinking
            )

            # 更新统计
            stats["total"] += 1
            stats["total_iterations"] += len(iteration_history)
            
            if text is None or (iteration_history and iteration_history[-1].get("error")):
                stats["prompt_too_long"] += 1
            elif kept_first:
                stats["kept_first_answer"] += 1
            else:
                stats["changed_answer"] += 1

            # 保存日志
            log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
            table_info = get_table_info(table)
            with open(log_path, "w", encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"分析师建议增强模式 - 问题 #{global_i}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Title: {title}\n")
                f.write(f"Question: {question}\n")
                f.write(f"Expected Answer: {answer}\n")
                f.write(f"Table Info: {table_info['rows']} rows x {table_info['cols']} cols, {table_info['char_length']} chars\n\n")
                f.write("-" * 60 + "\n")
                f.write("Table:\n")
                f.write(table + "\n")
                f.write("-" * 60 + "\n\n")
                
                for iter_info in iteration_history:
                    f.write(f"\n{'='*40}\n")
                    f.write(f"Round {iter_info['iteration']}\n")
                    f.write(f"{'='*40}\n\n")
                    
                    f.write("【Model Output】\n")
                    f.write(f"{iter_info.get('reasoning', 'N/A')}\n\n")
                    
                    f.write(f"【Extracted Answer】: {iter_info.get('answer', 'N/A')}\n\n")
                    
                    if 'suggestion' in iter_info:
                        f.write("【Advisor Suggestion】\n")
                        f.write(f"{iter_info['suggestion']}\n\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"Final Result: {'Kept original' if kept_first else 'Changed answer'}\n")
                f.write("=" * 60 + "\n")

            # 保存结果
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
                "kept_first_answer": kept_first,
                "first_answer": iteration_history[0].get("answer") if iteration_history else None,
                "final_answer": iteration_history[-1].get("answer") if iteration_history else None,
            }

            with open(os.path.join(log_dir, "result.jsonl"), "a", encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False)
                f.write("\n")

            global_i += 1
            pbar.update(1)

            # 更新进度条
            if stats["total"] > 0:
                kept_rate = stats["kept_first_answer"] / stats["total"] * 100
                pbar.set_postfix({
                    "坚持原答案": f"{kept_rate:.1f}%",
                    "修改答案": stats["changed_answer"],
                    "过长跳过": stats["prompt_too_long"]
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
    print(f"坚持原答案: {stats['kept_first_answer']} ({stats['kept_first_answer']/max(stats['total'],1)*100:.2f}%)")
    print(f"修改答案: {stats['changed_answer']} ({stats['changed_answer']/max(stats['total'],1)*100:.2f}%)")
    print(f"Prompt过长跳过: {stats['prompt_too_long']}")
    print(f"平均迭代次数: {stats['total_iterations']/max(stats['total'],1):.2f}")
    print("=" * 80)


if __name__ == "__main__":
    Fire(main)
