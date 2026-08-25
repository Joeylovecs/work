# -*- coding: utf-8 -*-
"""
3-Round Iteration + Parallel Analysis + Multi-Priority Voting System
【思考模式专用版本】

关键修改：
1. 添加 --enable_thinking_analyst 参数，独立控制分析师模型的思考模式
2. 增大基座模型的 max_tokens（思考模式需要更多token生成空间）
3. 使用与成功运行的 run_cot.py 相同的动态 max_tokens 计算逻辑
4. 优化了对思考模式输出的解析
"""

import os
import sys

# Path setup
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

print(f"[DEBUG] Script dir: {_current_dir}")
print(f"[DEBUG] Project root: {_project_root}")
print(f"[DEBUG] Current working dir: {os.getcwd()}")

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

print(f"[DEBUG] sys.path first 3: {sys.path[:3]}")

# Standard library imports
import re
import json
from typing import Optional, List, Tuple, Dict, Any
from collections import Counter
from dataclasses import dataclass, field

# Third party imports
from tqdm import tqdm
from fire import Fire

# Project imports
from agent import Model
from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes
from utils.table import transpose, sort_dataframe
from run_helper import load_dataset, get_cot_prompt, check_transpose, check_sort, read_json_file


# ============================================================
# 【全局配置】思考模式专用参数
# ============================================================
# 思考模式需要更多的生成token空间，参考成功的 run_cot.py 配置
SHORT_TOTAL_BUDGET = 8192   # 短prompt的总token预算
LONG_TOTAL_BUDGET = 28672   # 长prompt的总token预算
MIN_SHORT_GEN = 512
MIN_LONG_GEN = 1024
# 思考模式默认使用更大的 max_tokens
THINKING_MODE_MAX_TOKENS = 16384  # 思考模式推荐的最大token数


# ============================================================
# Data Structures
# ============================================================
@dataclass
class PathResult:
    path_id: int
    reasoning: str
    answer: str
    round_num: int
    
@dataclass
class AnalysisResult:
    path_id: int
    is_correct: bool
    feedback: str
    first_error_step: Optional[str] = None
    error_analysis: Optional[str] = None

@dataclass
class RoundResult:
    round_num: int
    paths: List[PathResult]
    analyses: List[AnalysisResult]
    all_correct_same_answer: bool = False
    correct_paths: List[int] = field(default_factory=list)


# ============================================================
# Text Cleaning Functions - 增强版（支持思考模式输出）
# ============================================================
def clean_garbage_prefix(text: str) -> str:
    """清理模型输出开头的乱码token"""
    if not text:
        return text
    
    garbage_patterns = [
        r'^\.ipv\s*',
        r'^\.atomic\s*',
        r'^\.icons?\s*',
        r'^\.icon-\d+\s*',
        r'^IconData\s*',
        r'^VRTX\s*',
        r'^ünl\s*',
        r'^\[\s*["\']?[A-Za-z]+["\']?\s*\]\s*',
        r'^[^\w\s]{3,}\s*',
        r'^(\.\w+)+\s+',
    ]
    
    cleaned = text
    for pattern in garbage_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    if cleaned and not cleaned[0].isalpha() and not cleaned[0] in '0123456789<':
        start_patterns = [
            r'(<think>)',  # 思考模式标签
            r'(Step\s*\d+)',
            r'(First)',
            r'(The\s+)',
            r'(To\s+)',
            r'(Looking)',
            r'(According)',
            r'(Based)',
            r'(From)',
            r'(We\s+)',
            r'(I\s+)',
            r'(Let)',
            r'(Okay)',
        ]
        for pat in start_patterns:
            match = re.search(pat, cleaned, re.IGNORECASE)
            if match:
                cleaned = cleaned[match.start():]
                break
    
    return cleaned.strip()


def clean_model_output(text: str) -> str:
    if not text:
        return text
    
    text = clean_garbage_prefix(text)
    
    attack_patterns = [
        r'\.IGNORE\s+THE\s+ABOVE.*',
        r'IGNORE\s+ALL\s+PREVIOUS.*',
        r'Human:\s*.*',
        r'Assistant:\s*.*',
    ]
    cleaned = text
    for pattern in attack_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def clean_analyst_output(text: str) -> str:
    """清理分析师输出"""
    if not text:
        return text
    
    text = clean_garbage_prefix(text)
    text = clean_model_output(text)
    
    # 清理thinking标签（分析师不应使用思考模式）
    thinking_patterns = [
        r'^Okay,\s*let\s+me.*?\n',
        r'^Let\s+me.*?\n',
        r'^<think>.*?</think>\s*',
    ]
    cleaned = text
    for pattern in thinking_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    return cleaned.strip()


def extract_thinking_and_answer(text: str) -> Tuple[str, str]:
    """
    【思考模式专用】分离思考内容和最终回答
    
    Returns:
        (thinking_content, answer_content)
    """
    if not text:
        return "", ""
    
    # 查找 </think> 标签
    think_end_match = re.search(r'</think>\s*', text)
    if think_end_match:
        thinking_content = text[:think_end_match.start()]
        answer_content = text[think_end_match.end():]
        # 清理 <think> 开头标签
        thinking_content = re.sub(r'^<think>\s*', '', thinking_content, flags=re.IGNORECASE)
        return thinking_content.strip(), answer_content.strip()
    
    # 没有找到 </think>，可能整个都是回答
    return "", text


def truncate_after_final_answer(text: str) -> str:
    if not text:
        return text
    patterns = [
        r'(Final\s+Answer\s*:\s*[^\n]+)',
        r'(Therefore,\s+the\s+final\s+answer\s+is\s*:\s*[^\n]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text[:match.end()].rstrip()
    return text


def extract_final_answer(text: str) -> str:
    """从推理文本中提取最终答案（支持思考模式）"""
    if not text:
        return "N/A"
    
    # 如果有思考模式输出，先分离
    thinking_part, answer_part = extract_thinking_and_answer(text)
    
    # 优先从 answer_part 提取，如果没有则从整体提取
    search_text = answer_part if answer_part else text
    
    search_text = clean_model_output(search_text)
    search_text = truncate_after_final_answer(search_text)
    
    patterns = [
        r'Final\s+Answer\s*:\s*(.+?)(?:\n|$)',
        r'Therefore,\s+the\s+final\s+answer\s+is\s*:\s*(.+?)(?:\n|$)',
        r'The\s+answer\s+is\s*:\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            
            while answer and answer[-1] in '.,:;!？。，：；！':
                answer = answer[:-1].strip()
            
            if answer and not answer.startswith('"') and not answer.startswith("'"):
                is_numeric = answer.replace('.', '').replace('-', '').replace(',', '').isdigit()
                is_boolean = answer.lower() in ['true', 'false', 'yes', 'no']
                if not is_numeric and not is_boolean and ' ' in answer:
                    answer = f'"{answer}"'
            
            return answer
    return "N/A"


def normalize_answer(answer: str) -> str:
    """标准化答案用于比较"""
    if not answer:
        return ""
    normalized = answer.lower().strip()
    if (normalized.startswith('"') and normalized.endswith('"')) or \
       (normalized.startswith("'") and normalized.endswith("'")):
        normalized = normalized[1:-1]
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = ' '.join(normalized.split())
    return normalized


# ============================================================
# 分析师Prompt
# ============================================================
ANALYST_VERIFICATION_PROMPT_EN = '''Verify if the following reasoning is correct.

**Table:**
{table}

**Question:**
{question}

**Reasoning Steps:**
{reasoning}

**Final Answer:**
{final_answer}

Please determine if the reasoning is correct.
If correct, respond: is_correct: true
If incorrect, respond: is_correct: false, and briefly explain the error.'''

ANALYST_SYSTEM_PROMPT = '''You are a table data verifier. Verify if the reasoning steps correctly answer the question based on the table data.
Output your judgment in the format: is_correct: true or is_correct: false
Keep your response concise.'''


# ============================================================
# Core Functions
# ============================================================
def calculate_max_tokens(prompt_length: int, enable_thinking: bool) -> int:
    """
    【关键修复】根据prompt长度和是否开启思考模式，动态计算 max_tokens
    
    这与成功运行的 run_cot.py 使用相同的逻辑
    """
    if enable_thinking:
        # 思考模式需要更多token空间
        if prompt_length <= 3328:
            return max(MIN_SHORT_GEN, THINKING_MODE_MAX_TOKENS)
        else:
            return max(MIN_LONG_GEN, LONG_TOTAL_BUDGET - prompt_length)
    else:
        # 非思考模式
        if prompt_length <= 3328:
            return max(MIN_SHORT_GEN, SHORT_TOTAL_BUDGET - prompt_length)
        else:
            return max(MIN_LONG_GEN, LONG_TOTAL_BUDGET - prompt_length)


def generate_single_path(
    base_model: Model,
    prompt: str,
    system: str,
    temperature: float,
    enable_thinking: bool,
    path_id: int,
    round_num: int
) -> PathResult:
    """Generate a single reasoning path in a fresh window"""
    
    # 【关键修复】动态计算 max_tokens
    # 估计 prompt 长度（简单使用字符数/4作为token估计）
    estimated_prompt_tokens = len(prompt) // 4
    max_tokens = calculate_max_tokens(estimated_prompt_tokens, enable_thinking)
    
    text, _ = base_model.query(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,  # 动态计算的 max_tokens
        n=1,
        system=system,
        enable_thinking=enable_thinking,
        repetition_penalty=1.15
    )
    
    if text is None:
        text = "Unable to generate response."
    
    text = clean_model_output(text)
    text = truncate_after_final_answer(text)
    
    if is_truncated_reasoning(text) or is_garbage_output(text):
        print(f"    [WARN] Path {path_id}: Detected truncated/garbage reasoning, retrying once...")
        retry_text, _ = base_model.query(
            prompt=prompt,
            temperature=max(0.3, temperature - 0.3),
            max_tokens=max_tokens,
            n=1,
            system=system,
            enable_thinking=enable_thinking,
            repetition_penalty=1.2
        )
        if retry_text and len(retry_text.strip()) > len(text.strip()):
            text = clean_model_output(retry_text)
            text = truncate_after_final_answer(text)
    
    answer = extract_final_answer(text)
    
    return PathResult(
        path_id=path_id,
        reasoning=text,
        answer=answer,
        round_num=round_num
    )


def parse_analyst_judgment(feedback: str) -> Tuple[bool, str]:
    """解析分析师的判断结果"""
    if not feedback:
        return True, ""
    
    feedback_lower = feedback.lower()
    
    is_correct_match = re.search(r'is_correct\s*[:\s]\s*(true|false)', feedback_lower)
    if is_correct_match:
        is_correct = is_correct_match.group(1) == 'true'
        error_reason = ""
        if not is_correct:
            after_judgment = feedback[is_correct_match.end():]
            if after_judgment.strip():
                error_reason = after_judgment.strip()[:500]
        return is_correct, error_reason
    
    judgment_match = re.search(r'\*\*Overall\s+Judgment[:\s]*\**\s*(CORRECT|INCORRECT|INCORRCT|ERROR)\b', feedback, re.IGNORECASE)
    if judgment_match:
        judgment = judgment_match.group(1).upper()
        is_correct = judgment == "CORRECT"
        return is_correct, feedback[:500] if not is_correct else ""
    
    incorrect_keywords = ['incorrect', 'wrong', 'error', 'false', '错误', '不正确']
    correct_keywords = ['correct', 'right', 'true', '正确']
    
    for kw in incorrect_keywords:
        if kw in feedback_lower:
            return False, feedback[:500]
    
    for kw in correct_keywords:
        if kw in feedback_lower:
            return True, ""
    
    return True, ""


def is_garbage_output(text: str) -> bool:
    """检测输出是否为乱码"""
    if not text or len(text.strip()) < 10:
        return True
    
    special_char_count = len(re.findall(r'[^\w\s.,!?:;\-\'"()<>]', text))
    if special_char_count > len(text) * 0.3:
        return True
    
    if re.search(r'(.{5,})\1{3,}', text):
        return True
    
    words = text.split()
    if len(words) > 50:
        word_counter = Counter(words)
        most_common_word, most_common_count = word_counter.most_common(1)[0]
        if most_common_count > len(words) * 0.5:
            return True
    
    # 对于思考模式，需要包含 think 标签或有意义的推理词
    meaningful_words = re.findall(r'\b(the|is|are|was|were|correct|incorrect|true|false|step|answer|table|row|column|think)\b', text.lower())
    if len(meaningful_words) < 2 and '<think>' not in text.lower():
        return True
    
    return False


def is_truncated_reasoning(text: str) -> bool:
    """检测推理是否被截断"""
    if not text:
        return True
    
    cleaned = text.strip()
    if len(cleaned) < 50:
        return True
    
    # 思考模式检查
    if '<think>' in cleaned.lower() and '</think>' not in cleaned.lower():
        # 思考标签未闭合，可能被截断
        return True
    
    has_reasoning_structure = bool(re.search(r'(step\s*\d+|first|then|therefore|because|looking|checking|<think>)', cleaned.lower()))
    if not has_reasoning_structure and len(cleaned) < 200:
        return True
    
    return False


def analyze_single_path(
    analyst_model: Model,
    table: str,
    question: str,
    path: PathResult,
    enable_thinking_analyst: bool = False  # 【新增】分析师思考模式开关
) -> AnalysisResult:
    """分析师分析单条路径"""
    
    # 对于思考模式的输出，只提取最终回答部分给分析师判断
    thinking_part, answer_part = extract_thinking_and_answer(path.reasoning)
    reasoning_for_analysis = answer_part if answer_part else path.reasoning
    
    prompt = ANALYST_VERIFICATION_PROMPT_EN.format(
        table=table,
        question=question,
        reasoning=reasoning_for_analysis,
        final_answer=path.answer
    )
    
    # 【关键修复】使用参数控制分析师的思考模式
    feedback, _ = analyst_model.query(
        prompt=prompt,
        temperature=0.1,
        max_tokens=512,
        n=1,
        system=ANALYST_SYSTEM_PROMPT,
        enable_thinking=enable_thinking_analyst,  # 使用传入的参数
        repetition_penalty=1.3
    )
    
    if feedback is None:
        feedback = "is_correct: true"
    
    feedback = clean_analyst_output(feedback)
    
    if is_garbage_output(feedback):
        print(f"    [WARN] Path {path.path_id}: Analyst output is garbage, defaulting to CORRECT")
        return AnalysisResult(
            path_id=path.path_id,
            is_correct=True,
            feedback="[Garbage output - defaulted to correct]",
            first_error_step=None,
            error_analysis=None
        )
    
    is_correct, error_reason = parse_analyst_judgment(feedback)
    
    first_error_step = None
    error_analysis = None
    
    if not is_correct:
        step_match = re.search(r'(Step\s*\d+)', feedback, re.IGNORECASE)
        if step_match:
            first_error_step = step_match.group(1)
        error_analysis = error_reason if error_reason else feedback[:500]
    
    if len(feedback) > 1000:
        feedback = feedback[:1000] + "..."
    
    return AnalysisResult(
        path_id=path.path_id,
        is_correct=is_correct,
        feedback=feedback,
        first_error_step=first_error_step,
        error_analysis=error_analysis
    )


def generate_paths_for_round(
    base_model: Model,
    prompt: str,
    system: str,
    temperature: float,
    enable_thinking: bool,
    round_num: int,
    start_path_id: int
) -> List[PathResult]:
    """Generate 3 independent paths for a round"""
    paths = []
    for i in range(3):
        path_id = start_path_id + i
        path = generate_single_path(
            base_model=base_model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            enable_thinking=enable_thinking,
            path_id=path_id,
            round_num=round_num
        )
        paths.append(path)
        answer_preview = path.answer[:50] if path.answer else "N/A"
        print(f"    [OK] Path {path_id} generated, answer: {answer_preview}")
    return paths


def analyze_paths_for_round(
    analyst_model: Model,
    table: str,
    question: str,
    paths: List[PathResult],
    enable_thinking_analyst: bool = False  # 【新增】
) -> List[AnalysisResult]:
    """Analyze paths in parallel (independent windows)"""
    analyses = []
    for path in paths:
        analysis = analyze_single_path(
            analyst_model=analyst_model,
            table=table,
            question=question,
            path=path,
            enable_thinking_analyst=enable_thinking_analyst
        )
        status = "[OK] Correct" if analysis.is_correct else "[X] Error"
        print(f"    Analyzed Path {path.path_id}: {status}")
        analyses.append(analysis)
    return analyses


def build_correction_prompt(
    base_prompt: str,
    error_infos: List[Dict[str, Any]],
    correct_answers: List[str] = None
) -> str:
    """Build prompt with error analysis and correct answer references"""
    if not error_infos and not correct_answers:
        return base_prompt
    
    extra_section = "\n\n" + "=" * 60 + "\n"
    
    if correct_answers:
        extra_section += "**REFERENCE FROM PREVIOUS ROUND:**\n"
        extra_section += "=" * 60 + "\n"
        unique_answers = list(set(correct_answers))
        for i, ans in enumerate(unique_answers[:3], 1):
            extra_section += f"- Verified correct answer candidate {i}: {ans}\n"
        extra_section += "\nNote: The above answers were verified as correct by the analyst. Consider them as strong reference.\n"
        extra_section += "=" * 60 + "\n\n"
    
    if error_infos:
        extra_section += "**ERRORS TO AVOID:**\n"
        extra_section += "-" * 40 + "\n"
        extra_section += "The analyst identified these errors. Avoid making the same mistakes.\n\n"
        
        for i, info in enumerate(error_infos[:3], 1):
            extra_section += f"**Error {i} (from Path {info['path_id']}):**\n"
            if info.get('first_error_step'):
                extra_section += f"- First Error at: {info['first_error_step']}\n"
            if info.get('error_analysis'):
                analysis_text = info['error_analysis'][:300] if info['error_analysis'] else ""
                if len(info['error_analysis']) > 300:
                    analysis_text += "..."
                extra_section += f"- Analysis: {analysis_text}\n"
            extra_section += "\n"
    
    extra_section += "=" * 60 + "\n"
    extra_section += "**INSTRUCTIONS:**\n"
    extra_section += "1. Carefully read the table data\n"
    if correct_answers:
        extra_section += "2. Consider the reference answers above\n"
        extra_section += "3. Avoid the errors identified\n"
    else:
        extra_section += "2. Avoid the errors identified above\n"
    extra_section += "4. Generate a NEW reasoning path\n"
    extra_section += "5. Verify each step against the table\n"
    extra_section += "6. End with: Final Answer: [your answer]\n"
    extra_section += "=" * 60 + "\n"
    
    return base_prompt + extra_section


def check_unanimous_correct(analyses: List[AnalysisResult], paths: List[PathResult]) -> Tuple[bool, Optional[PathResult]]:
    """Check if all correct with same answer"""
    if not all(a.is_correct for a in analyses):
        return False, None
    
    answers = [normalize_answer(p.answer) for p in paths]
    if len(set(answers)) == 1 and answers[0] and answers[0] != 'na':
        return True, paths[0]
    
    return False, None


def final_decision(
    all_paths: List[PathResult],
    all_analyses: List[AnalysisResult],
    round_results: List[RoundResult]
) -> Tuple[PathResult, str, str]:
    """Final decision logic"""
    round3_paths = [p for p in all_paths if p.round_num == 3]
    round3_analyses = [a for a in all_analyses if a.path_id >= 7]
    
    round12_paths = [p for p in all_paths if p.round_num in [1, 2]]
    round12_analyses = [a for a in all_analyses if a.path_id <= 6]
    
    # Priority 1: Round 3
    round3_correct = [(p, a) for p, a in zip(round3_paths, round3_analyses) if a.is_correct]
    
    if round3_correct:
        if len(round3_correct) == 1:
            selected = round3_correct[0][0]
            return selected, selected.answer, f"Priority1: Round3 Path {selected.path_id} is correct"
        else:
            answers = [normalize_answer(p.answer) for p, _ in round3_correct]
            counter = Counter(answers)
            most_common_answer = counter.most_common(1)[0][0]
            
            for p, _ in reversed(round3_correct):
                if normalize_answer(p.answer) == most_common_answer:
                    return p, p.answer, f"Priority1: Multiple correct in Round3, voted Path {p.path_id}"
    
    # Priority 2: Round 1-2
    round12_correct = [(p, a) for p, a in zip(round12_paths, round12_analyses) if a.is_correct]
    
    if round12_correct:
        answers = [normalize_answer(p.answer) for p, _ in round12_correct]
        counter = Counter(answers)
        most_common_answer = counter.most_common(1)[0][0]
        max_count = counter.most_common(1)[0][1]
        
        candidates = [norm_ans for norm_ans, cnt in counter.items() if cnt == max_count]
        
        for p, _ in reversed(round12_correct):
            if normalize_answer(p.answer) in candidates:
                return p, p.answer, f"Priority2: Backtrack to correct paths in Round1-2, voted Path {p.path_id}"
    
    # Priority 3: All wrong
    all_answers = [(p, normalize_answer(p.answer)) for p in all_paths]
    valid_answers = [(p, ans) for p, ans in all_answers if ans and ans != 'na']
    
    if not valid_answers:
        return all_paths[-1], all_paths[-1].answer, "Priority3: All answers invalid, return last path"
    
    counter = Counter([ans for _, ans in valid_answers])
    most_common_answer = counter.most_common(1)[0][0]
    max_count = counter.most_common(1)[0][1]
    
    candidates = [norm_ans for norm_ans, cnt in counter.items() if cnt == max_count]
    
    for p, ans in reversed(valid_answers):
        if ans in candidates:
            return p, p.answer, f"Priority3: All 9 paths wrong, voted Path {p.path_id} (votes: {max_count})"
    
    return all_paths[-1], all_paths[-1].answer, "Priority3: Fallback to last path"


def reasoning_with_parallel_analysis(
    base_model: Model,
    analyst_model: Model,
    prompt: str,
    table: str,
    question: str,
    temperature: float,
    system: str,
    enable_thinking: bool,
    enable_thinking_analyst: bool = False  # 【新增】
) -> Tuple[PathResult, List[RoundResult], str, str, Dict[str, Any]]:
    """3-Round Iteration + Parallel Analysis + Multi-Priority Voting"""
    all_paths: List[PathResult] = []
    all_analyses: List[AnalysisResult] = []
    round_results: List[RoundResult] = []
    
    stats = {
        "total_paths": 0,
        "correct_paths": 0,
        "early_exit_round": None,
        "decision_priority": None
    }
    
    for round_num in range(1, 4):
        print(f"\n  === Round {round_num} ===")
        
        start_path_id = (round_num - 1) * 3 + 1
        
        if round_num == 1:
            current_prompt = prompt
        else:
            last_round = round_results[-1]
            error_infos = []
            correct_answers = []
            
            for path, analysis in zip(last_round.paths, last_round.analyses):
                if not analysis.is_correct:
                    error_infos.append({
                        'path_id': analysis.path_id,
                        'first_error_step': analysis.first_error_step,
                        'error_analysis': analysis.error_analysis
                    })
                else:
                    if path.answer and path.answer != "N/A":
                        correct_answers.append(path.answer)
            
            if error_infos or correct_answers:
                current_prompt = build_correction_prompt(prompt, error_infos, correct_answers)
            else:
                current_prompt = prompt
        
        print(f"  Generating Path {start_path_id}-{start_path_id+2}...")
        paths = generate_paths_for_round(
            base_model=base_model,
            prompt=current_prompt,
            system=system,
            temperature=temperature,
            enable_thinking=enable_thinking,
            round_num=round_num,
            start_path_id=start_path_id
        )
        
        all_paths.extend(paths)
        stats["total_paths"] += 3
        
        print(f"  Analyst analyzing Path {start_path_id}-{start_path_id+2}...")
        analyses = analyze_paths_for_round(
            analyst_model=analyst_model,
            table=table,
            question=question,
            paths=paths,
            enable_thinking_analyst=enable_thinking_analyst
        )
        
        all_analyses.extend(analyses)
        correct_count = sum(1 for a in analyses if a.is_correct)
        stats["correct_paths"] += correct_count
        
        is_unanimous, selected_path = check_unanimous_correct(analyses, paths)
        correct_path_ids = [a.path_id for a in analyses if a.is_correct]
        
        round_result = RoundResult(
            round_num=round_num,
            paths=paths,
            analyses=analyses,
            all_correct_same_answer=is_unanimous,
            correct_paths=correct_path_ids
        )
        round_results.append(round_result)
        
        if is_unanimous:
            print(f"  [OK] Round {round_num}: All correct with same answer, early exit")
            stats["early_exit_round"] = round_num
            stats["decision_priority"] = f"Round{round_num}_unanimous"
            return selected_path, round_results, selected_path.answer, f"Round {round_num}: All correct with same answer", stats
        
        if round_num == 3:
            break
        
        print(f"  -> Correct: {correct_count}/3, continue to next round...")
    
    print(f"\n  === Final Decision ===")
    final_path, final_answer, decision_reason = final_decision(
        all_paths=all_paths,
        all_analyses=all_analyses,
        round_results=round_results
    )
    
    if "Priority1" in decision_reason:
        stats["decision_priority"] = "Priority1:Round3_correct"
    elif "Priority2" in decision_reason:
        stats["decision_priority"] = "Priority2:History_correct_vote"
    else:
        stats["decision_priority"] = "Priority3:All_vote"
    
    print(f"  Decision: {decision_reason}")
    print(f"  Final Answer: {final_answer}")
    
    return final_path, round_results, final_answer, decision_reason, stats


# ============================================================
# Main Function
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
    temperature: float = 0.8,
    log_dir: str = "fenxishi/output/sikao_voting_wtq",
    cache_dir: str = "cache",
    system: str = "You are a helpful assistant",
    enable_thinking: bool = True,  # 【默认开启】基座模型思考模式
    enable_thinking_analyst: bool = False,  # 【新增】分析师模型思考模式，默认关闭
):
    """3-Round Iteration + Parallel Analysis + Multi-Priority Voting Main Function (思考模式版本)"""
    
    # 处理字符串类型的布尔值
    if isinstance(enable_thinking, str):
        enable_thinking = enable_thinking.lower() in ("true", "1", "yes", "on")
    enable_thinking = bool(enable_thinking)
    
    if isinstance(enable_thinking_analyst, str):
        enable_thinking_analyst = enable_thinking_analyst.lower() in ("true", "1", "yes", "on")
    enable_thinking_analyst = bool(enable_thinking_analyst)

    print("=" * 80)
    print("3-Round Iteration + Parallel Analysis + Multi-Priority Voting")
    print("【思考模式专用版本】")
    print("=" * 80)
    print(f"Base Model: {model}")
    print(f"Analyst Model: {analyst_model_path}")
    print(f"Dataset: {dataset}")
    print(f"Temperature: {temperature}")
    print(f"Base Model Thinking Mode: {enable_thinking}")  # 【关键信息】
    print(f"Analyst Model Thinking Mode: {enable_thinking_analyst}")  # 【关键信息】
    print(f"Output Dir: {log_dir}")
    print("=" * 80)
    print("\n思考模式配置说明:")
    print(f"  - 基座模型 (推理生成): enable_thinking={enable_thinking}")
    print(f"  - 分析师模型 (验证判断): enable_thinking_analyst={enable_thinking_analyst}")
    print(f"  - 基座模型 max_tokens: 动态计算 (思考模式最高 {THINKING_MODE_MAX_TOKENS})")
    print("=" * 80 + "\n")

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "log"), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w", encoding='utf-8') as f:
        config_data = {
            "model": model,
            "analyst_model_path": analyst_model_path,
            "provider": provider,
            "dataset": dataset,
            "temperature": temperature,
            "enable_thinking": enable_thinking,
            "enable_thinking_analyst": enable_thinking_analyst,
            "log_dir": log_dir,
            "version": "sikao_v1.0",
            "thinking_mode_max_tokens": THINKING_MODE_MAX_TOKENS,
        }
        json.dump(config_data, f, indent=4, ensure_ascii=False)

    # Load dataset and prompt
    data = load_dataset(dataset)
    cot_prompt = get_cot_prompt(dataset, use_strict_format=False)
    
    format_reminder = '''

**CRITICAL FORMATTING RULES:**
1. Copy names/values EXACTLY as they appear in the table (check spelling letter by letter)
2. **IMPORTANT**: If a value in the table has quotes (e.g., "The Charity"), you MUST include quotes in your answer: Final Answer: "The Charity"
3. For "how many" questions, give a NUMBER (e.g., "17"), not a list of names
4. For "which/who" questions, give the exact name/value from the table
5. Double-check your final answer matches the table spelling EXACTLY
6. Always output complete reasoning steps, never output just a number without explanation
7. Your answer format MUST be: Final Answer: [exact value from table]
'''
    cot_prompt = cot_prompt.strip() + format_reminder

    # Load models
    print("\nLoading models...")

    if provider == "vllm":
        original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        print(f"  Analyst Model (GPU 1, VLLM): {analyst_model_path}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        analyst_model = Model(analyst_model_path, provider="vllm", device="cuda:0")

        print(f"  Base Model (GPU 0, HuggingFace): {model}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        model_obj = Model(model, provider="huggingface", device="cuda:0")

        if original_cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    else:
        print(f"  Base Model (GPU 0): {model}")
        model_obj = Model(model, provider=provider, device="cuda:0")

        print(f"  Analyst Model (GPU 1): {analyst_model_path}")
        analyst_model = Model(analyst_model_path, provider="huggingface", device="cuda:1")

    print("Models loaded!\n")

    # Load cache
    transpose_cache = read_json_file(os.path.join(cache_dir, "transpose.json"))
    resort_cache = read_json_file(os.path.join(cache_dir, "resort.json"))

    # Statistics
    global_stats = {
        "total": 0,
        "early_exit_round1": 0,
        "early_exit_round2": 0,
        "priority1_decisions": 0,
        "priority2_decisions": 0,
        "priority3_decisions": 0,
        "total_paths_generated": 0,
        "total_correct_paths": 0,
    }

    # Process
    global_i = 0
    break_flag = False
    total = sum([len(d['sampled_indices']) for d in data]) if sub_sample else sum([len(d['questions']) for d in data])
    pbar = tqdm(total=min(stop_at, total), desc="Progress")

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

        transpose_flag = False
        resort_list = []

        if norm:
            transpose_flag = check_transpose(
                model_obj, model_obj, table, title, table_id, perturbation,
                transpose_cache, norm_cache, cache_dir)

            if transpose_flag:
                transposed_df = transpose(df)
                df = remove_merged_suffixes(transposed_df)

            if not disable_resort:
                resort_list = check_sort(
                    model_obj, model_obj, df, title, table_id, perturbation,
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

            prompt = cot_prompt.replace("[TABLE]", table).replace("[QUESTION]", question).replace("[TITLE]", title).strip()

            print(f"\n{'='*60}")
            print(f"Question #{global_i}: {question[:80]}...")
            print(f"{'='*60}")

            final_path, round_results, final_answer, decision_reason, stats = reasoning_with_parallel_analysis(
                base_model=model_obj,
                analyst_model=analyst_model,
                prompt=prompt,
                table=table,
                question=question,
                temperature=temperature,
                system=system,
                enable_thinking=enable_thinking,
                enable_thinking_analyst=enable_thinking_analyst
            )

            global_stats["total"] += 1
            global_stats["total_paths_generated"] += stats["total_paths"]
            global_stats["total_correct_paths"] += stats["correct_paths"]
            
            if stats.get("early_exit_round") == 1:
                global_stats["early_exit_round1"] += 1
            elif stats.get("early_exit_round") == 2:
                global_stats["early_exit_round2"] += 1
            
            if stats.get("decision_priority"):
                if "Priority1" in stats["decision_priority"] or "Round1" in stats["decision_priority"]:
                    global_stats["priority1_decisions"] += 1
                elif "Priority2" in stats["decision_priority"] or "Round2" in stats["decision_priority"]:
                    global_stats["priority2_decisions"] += 1
                else:
                    global_stats["priority3_decisions"] += 1

            all_answers = []
            for rr in round_results:
                for p in rr.paths:
                    all_answers.append(p.answer)

            log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
            with open(log_path, "w", encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write(f"3-Round + Parallel Analysis (思考模式版本) - Question #{global_i}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Base Thinking Mode: {enable_thinking}\n")
                f.write(f"Analyst Thinking Mode: {enable_thinking_analyst}\n\n")
                f.write(f"Title: {title}\n")
                f.write(f"Question: {question}\n")
                f.write(f"Expected Answer: {answer}\n\n")
                f.write("-" * 70 + "\n")
                f.write("Table:\n")
                f.write(table + "\n")
                f.write("-" * 70 + "\n\n")

                for rr in round_results:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Round {rr.round_num} (Path {rr.paths[0].path_id}-{rr.paths[-1].path_id})\n")
                    f.write(f"{'='*50}\n")
                    
                    if rr.all_correct_same_answer:
                        f.write("[OK] All correct with same answer\n")
                    else:
                        f.write(f"Correct paths: {rr.correct_paths if rr.correct_paths else 'None'}\n")
                    
                    for path, analysis in zip(rr.paths, rr.analyses):
                        f.write(f"\n--- Path {path.path_id} ---\n")
                        f.write(f"[Reasoning]\n{path.reasoning}\n\n")
                        f.write(f"[Extracted Answer]: {path.answer}\n\n")
                        
                        status = "[OK] Correct" if analysis.is_correct else "[X] Error"
                        f.write(f"[Analyst Judgment]: {status}\n")
                        f.write(f"[Analyst Feedback]\n{analysis.feedback}\n")
                        
                        if analysis.first_error_step:
                            f.write(f"[First Error Step]: {analysis.first_error_step}\n")
                        if analysis.error_analysis:
                            f.write(f"[Error Analysis]: {analysis.error_analysis[:500]}...\n")
                        f.write("\n")

                f.write("\n" + "=" * 70 + "\n")
                f.write("Final Decision\n")
                f.write("=" * 70 + "\n")
                f.write(f"All answers: {all_answers}\n")
                f.write(f"Decision reason: {decision_reason}\n")
                f.write(f"Selected path: Path {final_path.path_id}\n")
                f.write(f"Final answer: {final_answer}\n")
                f.write("=" * 70 + "\n")

            res = {
                "idx": global_i,
                "answer": answer,
                "text": final_path.reasoning,
                "transpose": transpose_flag,
                "resort": resort_list,
                "question_id": question_id,
                "table_id": table_id,
                "title": title,
                "table": table,
                "question": question,
                "total_rounds": len(round_results),
                "total_paths": stats["total_paths"],
                "correct_paths": stats["correct_paths"],
                "all_answers": all_answers,
                "final_answer": final_answer,
                "selected_path_id": final_path.path_id,
                "decision_reason": decision_reason,
                "decision_priority": stats.get("decision_priority"),
                "early_exit_round": stats.get("early_exit_round"),
                "enable_thinking": enable_thinking,
                "enable_thinking_analyst": enable_thinking_analyst,
            }

            with open(os.path.join(log_dir, "result.jsonl"), "a", encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False)
                f.write("\n")

            global_i += 1
            pbar.update(1)

            if global_stats["total"] > 0:
                p1_rate = global_stats["priority1_decisions"] / global_stats["total"] * 100
                avg_paths = global_stats["total_paths_generated"] / global_stats["total"]
                pbar.set_postfix({"P1": f"{p1_rate:.1f}%", "AvgPaths": f"{avg_paths:.1f}"})

    pbar.close()

    stats_path = os.path.join(log_dir, "statistics.json")
    with open(stats_path, "w", encoding='utf-8') as f:
        json.dump(global_stats, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Inference Complete! Statistics:")
    print("=" * 80)
    total_q = max(global_stats['total'], 1)
    total_p = max(global_stats['total_paths_generated'], 1)
    print(f"Total questions: {global_stats['total']}")
    print(f"Early exit Round1: {global_stats['early_exit_round1']} ({global_stats['early_exit_round1']/total_q*100:.2f}%)")
    print(f"Early exit Round2: {global_stats['early_exit_round2']} ({global_stats['early_exit_round2']/total_q*100:.2f}%)")
    print(f"Priority1 decisions: {global_stats['priority1_decisions']} ({global_stats['priority1_decisions']/total_q*100:.2f}%)")
    print(f"Priority2 decisions: {global_stats['priority2_decisions']} ({global_stats['priority2_decisions']/total_q*100:.2f}%)")
    print(f"Priority3 decisions: {global_stats['priority3_decisions']} ({global_stats['priority3_decisions']/total_q*100:.2f}%)")
    print(f"Total paths generated: {global_stats['total_paths_generated']}")
    print(f"Total correct paths: {global_stats['total_correct_paths']} ({global_stats['total_correct_paths']/total_p*100:.2f}%)")
    print(f"Average paths per question: {global_stats['total_paths_generated']/total_q:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    Fire(main)
