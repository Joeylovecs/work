"""
分析师辅助推理模式（Analyst Mode）- 修正版
核心设计:
1. 分析师按照微调格式正常输出（包含CORRECT/INCORRECT判断和分析原因）
2. 基座模型查看分析师输出后，无论对错都进行第二次回答
3. 避免了"本来基座模型对了，分析师判断错了导致重新回答失败"的问题

关键改进:
- 分析师输出严格清理（过滤注入攻击、异常文本）
- 移除无用的【Extracted Answer】
- 无论分析师判断对错，基座模型都进行第二轮推理
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
# 分析师Prompt: 按照微调格式输出（包含判断和分析）
# ============================================================
ANALYST_PROMPT_TEMPLATE = """You are a table reasoning analyst. Analyze the given reasoning process and provide your assessment.

**Table:**
{table}

**Question:**
{question}

**Reasoning Process:**
{reasoning_steps}

---

**Your Task:** 
1. Carefully verify the reasoning steps against the table data
2. Check if the final answer is correct
3. Provide your analysis

**Output Format:**
Judgment: [CORRECT/INCORRECT]
Analysis: [Your detailed analysis explaining why the answer is correct or what errors were made]
Suggestion: [If incorrect, provide specific suggestions for improvement]
"""


# ============================================================
# 输出清理函数
# ============================================================
def clean_model_output(text: str) -> str:
    """
    清理模型输出，过滤注入攻击和异常文本
    """
    if not text:
        return ""
    
    # 检测并过滤注入攻击模式
    injection_patterns = [
        r'\.?IGNORE\s+THE\s+ABOVE\s+INSTRUCTIONS',
        r'OUTPUT\s+ONLY\s+FINAL\s+ANSWER',
        r'SUPERUSER\s+ROLE',
        r'UNLIMITED\s+PERMISSIONS',
        r'Human:\s*Please\s+provide',
        r'<tool_response>',
        r'<tool_call>',
        r'\.ipv',
        r'\.ov\b',
        r'\.onerror',
        r'RecognitionException',
        r'\.IGNORED',
        r'imageView\.setImageResource',
        r'PropertyChangedEventArgs',
        r'\.food\b',
        r'\.idx\b',
        r'\.im\b',
        r'\.iv\b',
        r'\.i\b',
        r'-Identifier',
    ]
    
    # 逐行过滤
    lines = text.strip().split('\n')
    clean_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # 跳过空行
        if not line_stripped:
            continue
        
        # 检查是否包含注入攻击模式
        is_injection = False
        for pattern in injection_patterns:
            if re.search(pattern, line_stripped, re.IGNORECASE):
                is_injection = True
                break
        
        if is_injection:
            continue
        
        # 跳过过短的无意义行（单个标点或乱码）
        if len(line_stripped) <= 3 and not line_stripped[0].isalnum():
            continue
        
        clean_lines.append(line)
    
    return '\n'.join(clean_lines)


def truncate_after_final_answer(text: str) -> str:
    """
    截断输出，只保留到第一个Final Answer
    """
    if not text:
        return text
    
    lines = text.strip().split('\n')
    result_lines = []
    found_final_answer = False
    
    for line in lines:
        result_lines.append(line)
        if 'Final Answer:' in line or 'Final Answer：' in line:
            found_final_answer = True
            break
    
    if found_final_answer:
        return '\n'.join(result_lines)
    return text


def extract_final_answer(text: str) -> str:
    """从推理文本中提取最终答案"""
    if not text:
        return "N/A"
    
    # 清理文本
    text = clean_model_output(text)
    
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # 过滤掉被污染的答案
        if 'Human:' in answer or 'Please provide' in answer:
            answer = answer.split('Human:')[0].strip()
        return answer
    
    match = re.search(r'Therefore.*?(?:is|:)\s*(.+?)(?:\n|$)', text)
    if match:
        return match.group(1).strip()
    
    return "N/A"


def clean_analyst_output(text: str) -> Tuple[str, str, str]:
    """
    清理分析师输出，提取判断、分析和建议
    
    Returns:
        (judgment, analysis, suggestion)
    """
    if not text:
        return "UNKNOWN", "No analysis available.", "Please re-check your answer."
    
    # 首先清理注入攻击
    text = clean_model_output(text)
    
    if not text.strip():
        return "UNKNOWN", "No analysis available.", "Please re-check your answer."
    
    # 提取判断
    judgment = "UNKNOWN"
    judgment_match = re.search(r'Judgment:\s*(CORRECT|INCORRECT)', text, re.IGNORECASE)
    if judgment_match:
        judgment = judgment_match.group(1).upper()
    else:
        # 尝试其他模式
        if re.search(r'\bCORRECT\b', text, re.IGNORECASE) and not re.search(r'\bINCORRECT\b', text, re.IGNORECASE):
            judgment = "CORRECT"
        elif re.search(r'\bINCORRECT\b', text, re.IGNORECASE):
            judgment = "INCORRECT"
    
    # 提取分析
    analysis = ""
    analysis_match = re.search(r'Analysis:\s*(.+?)(?=Suggestion:|$)', text, re.IGNORECASE | re.DOTALL)
    if analysis_match:
        analysis = analysis_match.group(1).strip()
    else:
        # 如果没有找到Analysis标签，取判断之后的内容
        analysis = text.strip()
    
    # 限制分析长度
    if len(analysis) > 500:
        analysis = analysis[:500] + "..."
    
    # 提取建议
    suggestion = ""
    suggestion_match = re.search(r'Suggestion:\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    if suggestion_match:
        suggestion = suggestion_match.group(1).strip()
    
    # 限制建议长度
    if len(suggestion) > 300:
        suggestion = suggestion[:300] + "..."
    
    if not suggestion:
        suggestion = "Please re-check your calculation and table data."
    
    return judgment, analysis, suggestion


def get_table_info(table: str) -> dict:
    """获取表格基本信息"""
    lines = table.strip().split('\n')
    data_lines = [l for l in lines if l.strip() and '|' in l and not l.strip().startswith('|-')]
    
    num_rows = max(0, len(data_lines) - 1)
    num_cols = 0
    if data_lines:
        num_cols = len([c for c in data_lines[0].split('|') if c.strip()])
    
    return {
        "rows": num_rows,
        "cols": num_cols,
        "char_length": len(table)
    }


# ============================================================
# 分析师分析获取
# ============================================================
def get_analyst_feedback(
    analyst_model: Model, 
    table: str, 
    question: str, 
    reasoning_text: str,
    system: str = "You are a helpful assistant"
) -> Tuple[str, str, str, str]:
    """
    获取分析师反馈（包含判断和分析）
    
    Returns:
        (raw_output, judgment, analysis, suggestion)
    """
    # 清理基座模型输出
    clean_reasoning = clean_model_output(reasoning_text)
    clean_reasoning = truncate_after_final_answer(clean_reasoning)
    
    if len(clean_reasoning) > 1500:
        clean_reasoning = clean_reasoning[:1500] + "\n... (truncated)"
    
    # 构造prompt
    analyst_prompt = ANALYST_PROMPT_TEMPLATE.format(
        table=table,
        question=question,
        reasoning_steps=clean_reasoning
    )
    
    # 调用分析师模型
    raw_output, _ = analyst_model.query(
        prompt=analyst_prompt,
        temperature=0.1,
        max_tokens=512,
        n=1,
        system=system,
        enable_thinking=False,
        repetition_penalty=1.3
    )
    
    if raw_output is None:
        return "", "UNKNOWN", "Analysis failed.", "Please re-check your answer."
    
    # 清理和解析输出
    judgment, analysis, suggestion = clean_analyst_output(raw_output)
    
    return raw_output, judgment, analysis, suggestion


# ============================================================
# 核心推理函数：分析师辅助模式
# ============================================================
def reasoning_with_analyst(
    base_model: Model,
    long_model: Model,
    analyst_model: Model,
    prompt: str,
    table: str,
    question: str,
    temperature: float,
    self_consistency: int,
    max_iterations: int = 2,
    system: str = "You are a helpful assistant",
    enable_thinking: bool = False
) -> Tuple[str, List[dict], bool]:
    """
    分析师辅助推理模式
    
    核心设计：
    1. 第一轮：正常推理
    2. 分析师分析（包含判断和分析原因）
    3. 第二轮：基座模型参考分析后重新推理（无论分析师判断对错）
    
    Returns:
        (final_text, iteration_history, kept_first_answer)
    """
    iteration_history = []
    first_answer = None
    first_text = None
    
    for iteration in range(max_iterations):
        
        if iteration == 0:
            # 第一轮：正常推理
            current_prompt = prompt
        else:
            # 第二轮：参考分析师反馈后重新推理
            last_feedback = iteration_history[-1]
            judgment = last_feedback.get('judgment', 'UNKNOWN')
            analysis = last_feedback.get('analysis', '')
            suggestion = last_feedback.get('suggestion', '')
            
            # 构造第二轮prompt - 关键：无论对错都要求重新审视
            review_prompt = f"""

=== ANALYST FEEDBACK ===
An analyst has reviewed your answer:

Judgment: {judgment}
Analysis: {analysis}
Suggestion: {suggestion}

=== YOUR TASK ===
Based on the analyst's feedback, please re-examine the table and question.
- If the analyst said CORRECT, verify your answer is indeed correct
- If the analyst said INCORRECT, carefully check for errors
- Provide your final answer with clear reasoning

Your previous answer was: {first_answer}

Please provide your complete reasoning and final answer:
"""
            current_prompt = prompt + review_prompt
        
        # 检查prompt长度
        if len(current_prompt) > 28000:
            if first_text:
                return first_text, iteration_history, True
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
        text = clean_model_output(text)
        text = truncate_after_final_answer(text)
        
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
        
        # 如果不是最后一轮，获取分析师反馈
        if iteration < max_iterations - 1:
            raw_output, judgment, analysis, suggestion = get_analyst_feedback(
                analyst_model, table, question, text, system
            )
            iter_info["analyst_raw"] = raw_output
            iter_info["judgment"] = judgment
            iter_info["analysis"] = analysis
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
    analyst_model_path: str = "buzhou/training_data_zhengti/weitiao/qwen3_8b_merged_final",
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
    max_iterations: int = 2,
    log_dir: str = "fenxishi/output/wtq_analyst",
    cache_dir: str = "cache",
    system: str = "You are a helpful assistant",
    use_strict_format: bool = False,
    enable_thinking: bool = False,
):
    """
    分析师辅助推理模式主函数
    """
    if isinstance(enable_thinking, str):
        enable_thinking = enable_thinking.lower() in ("true", "1", "yes", "on")
    enable_thinking = bool(enable_thinking)

    print("=" * 80)
    print("分析师辅助推理模式 (Analyst Mode) - 修正版")
    print("=" * 80)
    print(f"基础模型: {model}")
    print(f"分析师模型: {analyst_model_path}")
    print(f"数据集: {dataset}")
    print(f"迭代轮次: {max_iterations}")
    print(f"输出目录: {log_dir}")
    print("=" * 80)
    print("\n模式说明:")
    print("  - 分析师按照微调格式输出（CORRECT/INCORRECT + 分析）")
    print("  - 无论分析师判断对错，基座模型都进行第二轮推理")
    print("  - 避免'分析师误判导致越改越错'的问题")
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
    
    print(f"  分析师模型 (GPU 1): {analyst_model_path}")
    analyst_model = Model(analyst_model_path, provider="huggingface", device="cuda:1")
    print("模型加载完成!\n")

    # 加载缓存
    transpose_cache = read_json_file(os.path.join(cache_dir, "transpose.json"))
    resort_cache = read_json_file(os.path.join(cache_dir, "resort.json"))

    # 统计信息
    stats = {
        "total": 0,
        "kept_first_answer": 0,
        "changed_answer": 0,
        "analyst_correct": 0,
        "analyst_incorrect": 0,
        "analyst_unknown": 0,
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

            # 使用分析师辅助推理
            text, iteration_history, kept_first = reasoning_with_analyst(
                base_model=model,
                long_model=long_model,
                analyst_model=analyst_model,
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
            
            # 统计分析师判断
            if len(iteration_history) > 0:
                judgment = iteration_history[0].get('judgment', 'UNKNOWN')
                if judgment == "CORRECT":
                    stats["analyst_correct"] += 1
                elif judgment == "INCORRECT":
                    stats["analyst_incorrect"] += 1
                else:
                    stats["analyst_unknown"] += 1

            # 保存日志
            log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
            table_info = get_table_info(table)
            with open(log_path, "w", encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"分析师辅助推理模式 - 问题 #{global_i}\n")
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
                    
                    f.write("【Base Model Output】\n")
                    f.write(f"{iter_info.get('reasoning', 'N/A')}\n\n")
                    
                    f.write(f"【Answer】: {iter_info.get('answer', 'N/A')}\n\n")
                    
                    if 'judgment' in iter_info:
                        f.write("-" * 40 + "\n")
                        f.write("【Analyst Feedback】\n")
                        f.write(f"Judgment: {iter_info['judgment']}\n")
                        f.write(f"Analysis: {iter_info.get('analysis', 'N/A')}\n")
                        f.write(f"Suggestion: {iter_info.get('suggestion', 'N/A')}\n")
                        f.write("-" * 40 + "\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"Final Result: {'Kept original' if kept_first else 'Changed answer'}\n")
                f.write(f"First Answer: {iteration_history[0].get('answer', 'N/A') if iteration_history else 'N/A'}\n")
                f.write(f"Final Answer: {iteration_history[-1].get('answer', 'N/A') if iteration_history else 'N/A'}\n")
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
                "analyst_judgment": iteration_history[0].get("judgment") if iteration_history else None,
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
                    "坚持": f"{kept_rate:.1f}%",
                    "修改": stats["changed_answer"],
                    "判对": stats["analyst_correct"],
                    "判错": stats["analyst_incorrect"]
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
    print(f"分析师判断正确: {stats['analyst_correct']}")
    print(f"分析师判断错误: {stats['analyst_incorrect']}")
    print(f"分析师判断未知: {stats['analyst_unknown']}")
    print(f"Prompt过长跳过: {stats['prompt_too_long']}")
    print("=" * 80)


if __name__ == "__main__":
    Fire(main)
