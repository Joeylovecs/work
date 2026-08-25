# -*- coding: utf-8 -*-
"""
3-Round Iteration + Parallel Analysis + Multi-Priority Voting System

修复版本 v2.2 - 修复了以下问题:
1. 分析师prompt与训练格式匹配
2. 调整了生成参数 (temperature, repetition_penalty)
3. 改进了输出解析逻辑
4. 增加了对乱码输出的清理和容错
5. [NEW] 循环检测 (detect_repetition_loop) - 防止无限重复
6. [NEW] 分析师自我一致性 (n_votes=3) - 提高判断稳定性
7. [NEW] is_valid_format 标志 - 区分格式问题
8. [NEW] 增强格式提示 - 解决引号问题
9. [NEW] max_tokens 1024 - 避免基座模型截断
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
from run_helper import load_dataset, get_cot_prompt, query, check_transpose, check_sort, read_json_file


# ============================================================
# Data Structures
# ============================================================
@dataclass
class PathResult:
    path_id: int
    reasoning: str
    answer: str
    round_num: int
    is_valid_format: bool = True  # [NEW] 格式有效性标志
    
@dataclass
class AnalysisResult:
    path_id: int
    is_correct: bool
    feedback: str
    confidence_score: float = 0.5  # [NEW] 置信度分数
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
# Text Cleaning Functions - 增强版
# ============================================================

# [NEW] 循环检测函数 - 防止 log 55 类型的无限重复问题
def detect_repetition_loop(text: str, window_size: int = 20, threshold: int = 10) -> Tuple[bool, str]:
    """
    检测并处理输出中的重复循环
    
    Args:
        text: 输入文本
        window_size: 窗口大小 (词数)
        threshold: 重复阈值
        
    Returns:
        (is_loop_detected, processed_text)
    """
    if not text:
        return False, text
    
    words = text.split()
    if len(words) < window_size * 2:
        return False, text
    
    # 计算词汇多样性比率
    unique_ratio = len(set(words)) / len(words)
    
    # 如果多样性太低 (< 10%)，说明有严重重复
    if unique_ratio < 0.1:
        # 截断并标记
        truncated = ' '.join(words[:500])  # 保留前500词
        return True, truncated + "\n[Output Truncated: Repetition Loop Detected]"
    
    # 检查固定窗口重复模式
    for i in range(len(words) - window_size * 2):
        window = ' '.join(words[i:i + window_size])
        rest = ' '.join(words[i + window_size:])
        count = rest.count(window)
        if count >= threshold:
            # 找到重复，截断
            truncated = ' '.join(words[:i + window_size * 2])
            return True, truncated + f"\n[Output Truncated: Pattern '{window[:50]}...' repeated {count} times]"
    
    return False, text


def clean_garbage_prefix(text: str) -> str:
    """清理模型输出开头的乱码token
    
    常见乱码模式: .ipv, .atomic, .icons, IconData, ünl, VRTX等
    """
    if not text:
        return text
    
    # 常见乱码前缀模式
    garbage_patterns = [
        r'^\.ipv\s*',
        r'^\.atomic\s*',
        r'^\.icons?\s*',
        r'^\.icon-\d+\s*',
        r'^IconData\s*',
        r'^VRTX\s*',
        r'^ünl\s*',
        r'^\[\s*["\']?[A-Za-z]+["\']?\s*\]\s*',  # 如 ["paulO"]
        r'^[^\w\s]{3,}\s*',  # 连续3个以上特殊字符
        r'^(\.\w+)+\s+',  # 连续的 .xxx.yyy 模式
    ]
    
    cleaned = text
    for pattern in garbage_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # 如果开头还是有问题，尝试找到第一个有意义的英文单词
    if cleaned and not cleaned[0].isalpha() and not cleaned[0] in '0123456789':
        # 找到第一个以常见推理词开头的位置
        start_patterns = [
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
    
    # 首先清理乱码前缀
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
    """清理分析师输出，增强乱码处理"""
    if not text:
        return text
    
    # 清理乱码前缀
    text = clean_garbage_prefix(text)
    text = clean_model_output(text)
    
    # 清理thinking标签
    thinking_patterns = [
        r'^Okay,\s*let\s+me.*?\n',
        r'^Let\s+me.*?\n',
        r'^<think>.*?</think>\s*',
    ]
    cleaned = text
    for pattern in thinking_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    return cleaned.strip()


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
    if not text:
        return "N/A"
    text = clean_model_output(text)
    text = truncate_after_final_answer(text)
    patterns = [
        r'Final\s+Answer\s*:\s*(.+?)(?:\n|$)',
        r'Therefore,\s+the\s+final\s+answer\s+is\s*:\s*(.+?)(?:\n|$)',
        r'The\s+answer\s+is\s*:\s*(.+?)(?:\n|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            if (answer.startswith('"') and answer.endswith('"') and answer.count('"') == 2) or \
               (answer.startswith("'") and answer.endswith("'") and answer.count("'") == 2):
                inner = answer[1:-1]
                if '"' in inner or "'" in inner:
                    return answer.strip()
                answer = inner
            return answer.strip()
    return "N/A"


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    normalized = answer.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = ' '.join(normalized.split())
    return normalized


# ============================================================
# 【核心修复】分析师Prompt - 匹配训练数据格式
# ============================================================

# 新的分析师prompt - 简化版，匹配训练数据格式
ANALYST_VERIFICATION_PROMPT_SIMPLE = '''请验证以下推理过程是否正确。

**表格:**
{table}

**问题:**
{question}

**推理步骤:**
{reasoning}

**最终答案:**
{final_answer}

请判断推理是否正确。如果正确，回复: is_correct: true
如果错误，回复: is_correct: false，并简要说明错误原因。'''

# 备选英文版本 - 更接近训练格式
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

# 系统提示 - 简化版
ANALYST_SYSTEM_PROMPT = '''You are a table data verifier. Verify if the reasoning steps correctly answer the question based on the table data.
Output your judgment in the format: is_correct: true or is_correct: false
Keep your response concise.'''


# ============================================================
# Core Functions
# ============================================================

# [NEW] 基座模型最大token数 - 从10240保持，但添加循环检测
MAX_TOKENS_BASE = 1024  # 基座模型生成token数增加以避免截断

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
    text, _ = base_model.query(
        prompt=prompt,
        temperature=temperature,
        max_tokens=MAX_TOKENS_BASE,  # [CHANGED] 使用常量，便于调整
        n=1,
        system=system,
        enable_thinking=enable_thinking
    )
    
    if text is None:
        text = "Unable to generate response."
    
    # [NEW] 检测并处理重复循环
    is_loop, text = detect_repetition_loop(text)
    if is_loop:
        print(f"    [WARN] Path {path_id}: Repetition loop detected, output truncated")
    
    # 清理乱码和多余内容
    text = clean_model_output(text)
    text = truncate_after_final_answer(text)
    answer = extract_final_answer(text)
    
    # [NEW] 判断格式是否有效
    is_valid = answer != "N/A" and not is_loop
    
    return PathResult(
        path_id=path_id,
        reasoning=text,
        answer=answer,
        round_num=round_num,
        is_valid_format=is_valid
    )


def parse_analyst_judgment(feedback: str) -> Tuple[bool, str]:
    """解析分析师的判断结果
    
    优先查找 is_correct: true/false 格式
    fallback 到其他判断模式
    
    返回: (is_correct, error_reason)
    """
    if not feedback:
        return True, ""  # 默认正确
    
    feedback_lower = feedback.lower()
    
    # 优先级1: 查找 is_correct: true/false 格式 (匹配训练数据)
    is_correct_match = re.search(r'is_correct\s*[:\s]\s*(true|false)', feedback_lower)
    if is_correct_match:
        is_correct = is_correct_match.group(1) == 'true'
        # 提取错误原因 (如果有)
        error_reason = ""
        if not is_correct:
            # 查找 is_correct: false 之后的内容
            after_judgment = feedback[is_correct_match.end():]
            if after_judgment.strip():
                error_reason = after_judgment.strip()[:500]
        return is_correct, error_reason
    
    # 优先级2: 查找 Overall Judgment 格式 (向后兼容)
    judgment_match = re.search(r'\*\*Overall\s+Judgment[:\s]*\**\s*(CORRECT|INCORRECT|INCORRCT|ERROR)\b', feedback, re.IGNORECASE)
    if judgment_match:
        judgment = judgment_match.group(1).upper()
        is_correct = judgment == "CORRECT"
        return is_correct, feedback[:500] if not is_correct else ""
    
    # 优先级3: 关键词检测
    incorrect_keywords = ['incorrect', 'wrong', 'error', 'false', '错误', '不正确']
    correct_keywords = ['correct', 'right', 'true', '正确']
    
    # 先检查是否有否定关键词
    for kw in incorrect_keywords:
        if kw in feedback_lower:
            return False, feedback[:500]
    
    # 再检查是否有肯定关键词
    for kw in correct_keywords:
        if kw in feedback_lower:
            return True, ""
    
    # 默认正确 (避免误杀)
    return True, ""


def is_garbage_output(text: str) -> bool:
    """检测输出是否为乱码/无意义内容"""
    if not text or len(text.strip()) < 10:
        return True
    
    # 检查是否包含过多特殊字符
    special_char_count = len(re.findall(r'[^\w\s.,!?:;\-\'"()]', text))
    if special_char_count > len(text) * 0.3:
        return True
    
    # 检查是否有大量重复模式
    if re.search(r'(.{5,})\1{3,}', text):  # 同样内容重复3次以上
        return True
    
    # 检查是否包含有意义的词
    meaningful_words = re.findall(r'\b(the|is|are|was|were|correct|incorrect|true|false|step|answer|table|row|column)\b', text.lower())
    if len(meaningful_words) < 2:
        return True
    
    return False


def analyze_single_path(
    analyst_model: Model,
    table: str,
    question: str,
    path: PathResult
) -> AnalysisResult:
    """分析师分析单条路径 - 使用简化prompt"""
    
    # 【关键修复】使用匹配训练格式的prompt
    prompt = ANALYST_VERIFICATION_PROMPT_EN.format(
        table=table,
        question=question,
        reasoning=path.reasoning,
        final_answer=path.answer
    )
    
    # 【关键修复】调整生成参数
    feedback, _ = analyst_model.query(
        prompt=prompt,
        temperature=0.1,        # 从 0.0 改为 0.1，避免退化输出
        max_tokens=512,         # 从 1024 降低，判断应该简短
        n=1,
        system=ANALYST_SYSTEM_PROMPT,
        enable_thinking=False,
        repetition_penalty=1.2  # 从 2.0 降低到 1.2，减少干扰
    )
    
    if feedback is None:
        feedback = "is_correct: true"
    
    feedback = clean_analyst_output(feedback)
    
    # 检测乱码输出 - 如果是乱码则默认为正确 (避免误杀)
    if is_garbage_output(feedback):
        print(f"    [WARN] Path {path.path_id}: Analyst output is garbage, defaulting to CORRECT")
        return AnalysisResult(
            path_id=path.path_id,
            is_correct=True,
            feedback="[Garbage output - defaulted to correct]",
            confidence_score=0.3,  # [NEW] 低置信度
            first_error_step=None,
            error_analysis=None
        )
    
    # 解析判断结果
    is_correct, error_reason = parse_analyst_judgment(feedback)
    
    # 提取错误信息
    first_error_step = None
    error_analysis = None
    
    if not is_correct:
        # 尝试提取错误步骤
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
        confidence_score=1.0 if is_correct else 0.5,  # [NEW] 基础置信度
        first_error_step=first_error_step,
        error_analysis=error_analysis
    )


# [NEW] 分析师自我一致性投票函数
def analyze_path_with_consistency(
    analyst_model: Model,
    table: str,
    question: str,
    path: PathResult,
    n_votes: int = 3
) -> AnalysisResult:
    """
    分析师自我一致性投票机制
    
    对同一路径进行n次验证投票，取多数结果作为最终判断。
    这样可以减少分析师单次判断的不稳定性。
    
    Args:
        analyst_model: 分析师模型
        table: 表格数据
        question: 问题
        path: 待验证的推理路径
        n_votes: 投票次数 (默认3次)
        
    Returns:
        AnalysisResult with confidence score based on vote agreement
    """
    # 如果路径本身就是无效格式，直接返回错误
    if not path.is_valid_format:
        return AnalysisResult(
            path_id=path.path_id,
            is_correct=False,
            feedback="[Invalid format - answer not extracted]",
            confidence_score=0.0,
            first_error_step="Format",
            error_analysis="Could not extract valid answer from reasoning"
        )
    
    # 构建prompt
    prompt = ANALYST_VERIFICATION_PROMPT_EN.format(
        table=table,
        question=question,
        reasoning=path.reasoning,
        final_answer=path.answer
    )
    
    votes = []
    feedbacks = []
    check_temperature = 0.4  # [NEW] 投票时使用略高温度以获得多样性
    
    for vote_i in range(n_votes):
        feedback, _ = analyst_model.query(
            prompt=prompt,
            temperature=check_temperature,
            max_tokens=512,
            n=1,
            system=ANALYST_SYSTEM_PROMPT,
            enable_thinking=False,
            repetition_penalty=1.2
        )
        
        if feedback is None:
            continue
            
        feedback = clean_analyst_output(feedback)
        
        # 跳过乱码输出
        if is_garbage_output(feedback):
            continue
        
        # 解析判断
        is_correct_match = re.search(r'is_correct\s*[:\s]\s*(true|false)', feedback.lower())
        if is_correct_match:
            votes.append(is_correct_match.group(1) == 'true')
            feedbacks.append(feedback)
    
    # 计算投票结果
    if not votes:
        # 没有有效投票，默认正确但低置信度
        return AnalysisResult(
            path_id=path.path_id,
            is_correct=True,
            feedback="[No valid votes - defaulted to correct]",
            confidence_score=0.3,
            first_error_step=None,
            error_analysis=None
        )
    
    true_count = sum(votes)
    total_valid = len(votes)
    
    # 多数投票决定最终结果
    final_judgment = true_count > (total_valid / 2)
    
    # 置信度 = 多数票比例
    if final_judgment:
        confidence = true_count / total_valid
    else:
        confidence = (total_valid - true_count) / total_valid
    
    # 选择与最终判断一致的反馈
    selected_feedback = ""
    for vote, fb in zip(votes, feedbacks):
        if vote == final_judgment:
            selected_feedback = fb
            break
    
    if not selected_feedback and feedbacks:
        selected_feedback = feedbacks[0]
    
    # 提取错误信息
    first_error_step = None
    error_analysis = None
    
    if not final_judgment:
        step_match = re.search(r'(Step\s*\d+)', selected_feedback, re.IGNORECASE)
        if step_match:
            first_error_step = step_match.group(1)
        
        # 找错误原因
        after_false = re.search(r'is_correct\s*:\s*false[,.\s]*(.*)', selected_feedback, re.IGNORECASE | re.DOTALL)
        if after_false:
            error_analysis = after_false.group(1).strip()[:500]
        else:
            error_analysis = selected_feedback[:500]
    
    return AnalysisResult(
        path_id=path.path_id,
        is_correct=final_judgment,
        feedback=f"[Consistency Vote: {true_count}/{total_valid}] " + selected_feedback[:800],
        confidence_score=confidence,
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
        # 显示答案的前50个字符，方便调试
        answer_preview = path.answer[:50] if path.answer else "N/A"
        print(f"    [OK] Path {path_id} generated, answer: {answer_preview}")
    return paths


def analyze_paths_for_round(
    analyst_model: Model,
    table: str,
    question: str,
    paths: List[PathResult],
    use_consistency: bool = True,  # [NEW] 是否使用自我一致性
    n_votes: int = 3               # [NEW] 投票次数
) -> List[AnalysisResult]:
    """Analyze paths with optional self-consistency voting"""
    analyses = []
    for path in paths:
        if use_consistency:
            # [NEW] 使用自我一致性投票
            analysis = analyze_path_with_consistency(
                analyst_model=analyst_model,
                table=table,
                question=question,
                path=path,
                n_votes=n_votes
            )
        else:
            # 单次分析 (快速模式)
            analysis = analyze_single_path(
                analyst_model=analyst_model,
                table=table,
                question=question,
                path=path
            )
        status = "[OK] Correct" if analysis.is_correct else "[X] Error"
        conf_str = f"(Conf: {analysis.confidence_score:.2f})"
        print(f"    Analyzed Path {path.path_id}: {status} {conf_str}")
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
    
    # Add correct answer reference if available
    if correct_answers:
        extra_section += "**REFERENCE FROM PREVIOUS ROUND:**\n"
        extra_section += "=" * 60 + "\n"
        unique_answers = list(set(correct_answers))
        for i, ans in enumerate(unique_answers[:3], 1):
            extra_section += f"- Verified correct answer candidate {i}: {ans}\n"
        extra_section += "\nNote: The above answers were verified as correct by the analyst. Consider them as strong reference.\n"
        extra_section += "=" * 60 + "\n\n"
    
    # Add error information
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
    """Check if all valid paths are correct with same answer"""
    # [NEW] 只考虑格式有效的路径
    valid_indices = [i for i, p in enumerate(paths) if p.is_valid_format]
    if not valid_indices:
        return False, None
    
    valid_analyses = [analyses[i] for i in valid_indices]
    valid_paths = [paths[i] for i in valid_indices]
    
    if not all(a.is_correct for a in valid_analyses):
        return False, None
    
    answers = [normalize_answer(p.answer) for p in valid_paths]
    if len(set(answers)) == 1 and answers[0] and answers[0] != 'na':
        # [NEW] 选择置信度最高的路径
        best_idx = max(range(len(valid_analyses)), key=lambda i: valid_analyses[i].confidence_score)
        return True, valid_paths[best_idx]
    
    return False, None


def final_decision(
    all_paths: List[PathResult],
    all_analyses: List[AnalysisResult],
    round_results: List[RoundResult]
) -> Tuple[PathResult, str, str]:
    """
    Final decision logic with confidence scoring
    
    Priority 1: Round 3 (Path 7,8,9) has correct path (prefer higher confidence)
    Priority 2: Round 1-2 (Path 1-6) has correct path, vote among them
    Priority 3: All 9 wrong, vote all answers
    """
    # [NEW] 只考虑有效格式的路径
    round3_valid = [(p, all_analyses[p.path_id - 1]) for p in all_paths 
                    if p.round_num == 3 and p.is_valid_format]
    round12_valid = [(p, all_analyses[p.path_id - 1]) for p in all_paths 
                     if p.round_num in [1, 2] and p.is_valid_format]
    
    # Priority 1: Check Round 3 for correct paths
    round3_correct = [(p, a) for p, a in round3_valid if a.is_correct]
    
    if round3_correct:
        if len(round3_correct) == 1:
            selected = round3_correct[0][0]
            conf = round3_correct[0][1].confidence_score
            return selected, selected.answer, f"Priority1: Round3 Path {selected.path_id} is correct (Conf: {conf:.2f})"
        else:
            answers = [normalize_answer(p.answer) for p, _ in round3_correct]
            counter = Counter(answers)
            most_common_answer = counter.most_common(1)[0][0]
            
            for p, _ in reversed(round3_correct):
                if normalize_answer(p.answer) == most_common_answer:
                    return p, p.answer, f"Priority1: Multiple correct in Round3, voted Path {p.path_id}"
    
    # Priority 2: Check Round 1-2 for correct paths
    round12_correct = [(p, a) for p, a in round12_valid if a.is_correct]
    
    if round12_correct:
        answers = [normalize_answer(p.answer) for p, _ in round12_correct]
        counter = Counter(answers)
        most_common_answer = counter.most_common(1)[0][0]
        max_count = counter.most_common(1)[0][1]
        
        # [NEW] 如果有多个最高票，选择置信度更高的
        candidates = [(p, a) for p, a in round12_correct if normalize_answer(p.answer) == most_common_answer]
        candidates.sort(key=lambda x: x[1].confidence_score, reverse=True)
        selected = candidates[0][0]
        conf = candidates[0][1].confidence_score
        
        return selected, selected.answer, f"Priority2: Backtrack to correct paths in Round1-2, voted Path {selected.path_id} (Votes: {max_count}, Conf: {conf:.2f})"
    
    # Priority 3: All wrong, vote all valid-format answers
    all_valid = [(p, normalize_answer(p.answer)) for p in all_paths if p.is_valid_format]
    valid_answers = [(p, ans) for p, ans in all_valid if ans and ans != 'na']
    
    if not valid_answers:
        return all_paths[-1], all_paths[-1].answer, "Priority3: All answers invalid, return last path"
    
    counter = Counter([ans for _, ans in valid_answers])
    most_common_answer = counter.most_common(1)[0][0]
    max_count = counter.most_common(1)[0][1]
    
    # 选择最高票的最后一个路径 (通常来自更晚的轮次)
    for p, ans in reversed(valid_answers):
        if ans == most_common_answer:
            return p, p.answer, f"Priority3: All paths wrong, voted Path {p.path_id} (Votes: {max_count})"
    
    return all_paths[-1], all_paths[-1].answer, "Priority3: Fallback to last path"


# ============================================================
# Main Reasoning Function
# ============================================================
def reasoning_with_parallel_analysis(
    base_model: Model,
    analyst_model: Model,
    prompt: str,
    table: str,
    question: str,
    temperature: float,
    system: str,
    enable_thinking: bool
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
        
        # Build prompt
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
        
        # Generate 3 paths
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
        
        # Analyze 3 paths with self-consistency voting
        print(f"  Analyst analyzing Path {start_path_id}-{start_path_id+2} (Consistency Voting)...")
        analyses = analyze_paths_for_round(
            analyst_model=analyst_model,
            table=table,
            question=question,
            paths=paths,
            use_consistency=True,  # [NEW] 启用自我一致性投票
            n_votes=3              # [NEW] 每条路径3次投票
        )
        
        all_analyses.extend(analyses)
        correct_count = sum(1 for a in analyses if a.is_correct)
        stats["correct_paths"] += correct_count
        
        # Check unanimous correct
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
        
        # Early exit check
        if is_unanimous:
            print(f"  [OK] Round {round_num}: All correct with same answer, early exit")
            stats["early_exit_round"] = round_num
            stats["decision_priority"] = f"Round{round_num}_unanimous"
            return selected_path, round_results, selected_path.answer, f"Round {round_num}: All correct with same answer", stats
        
        if round_num == 3:
            break
        
        print(f"  -> Correct: {correct_count}/3, continue to next round...")
    
    # Final decision
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
    log_dir: str = "fenxishi/output/parallel_voting_wtq_fixed",
    cache_dir: str = "cache",
    system: str = "You are a helpful assistant",
    enable_thinking: bool = False,
):
    """3-Round Iteration + Parallel Analysis + Multi-Priority Voting Main Function (Fixed Version)"""
    if isinstance(enable_thinking, str):
        enable_thinking = enable_thinking.lower() in ("true", "1", "yes", "on")
    enable_thinking = bool(enable_thinking)

    print("=" * 80)
    print("3-Round Iteration + Parallel Analysis + Multi-Priority Voting (FIXED)")
    print("=" * 80)
    print(f"Base Model: {model}")
    print(f"Analyst Model: {analyst_model_path}")
    print(f"Dataset: {dataset}")
    print(f"Temperature: {temperature}")
    print(f"Thinking Mode: {enable_thinking}")
    print(f"Output Dir: {log_dir}")
    print("=" * 80)
    print("\n[v2.2 Enhancements Active]")
    print("  1. Simplified analyst prompt (matching training format)")
    print("  2. Analyst self-consistency voting (3 votes per path)")
    print("  3. Loop detection & safety (prevents infinite repetition)")
    print("  4. Garbage output detection and handling")
    print("  5. Enhanced output parsing (is_correct: true/false)")
    print("  6. is_valid_format flag for paths")
    print("  7. Confidence scoring in final decision")
    print("  8. Enhanced format prompts (quote handling)")
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
            "log_dir": log_dir,
            "version": "2.2",  # [UPDATED]
            "enhancements": [
                "simplified_analyst_prompt",
                "analyst_self_consistency_voting",
                "loop_detection",
                "garbage_output_detection",
                "is_correct_parsing",
                "is_valid_format_flag",
                "confidence_scoring",
                "enhanced_format_prompts"
            ]
        }
        json.dump(config_data, f, indent=4, ensure_ascii=False)

    # Load dataset and prompt
    data = load_dataset(dataset)
    cot_prompt = get_cot_prompt(dataset, use_strict_format=False)
    
    # [ENHANCED] COT prompt with stronger format matching instructions
    format_reminder = """

**CRITICAL FORMATTING RULES:**
1. Copy names/values EXACTLY as they appear in the table (check spelling letter by letter).
2. If a value in the table has quotes (e.g., "The Charity"), your answer MUST include the quotes.
3. For "how many" questions, give a NUMBER only (e.g., "5"), not a list of names.
4. For counting questions, first LIST each item explicitly, then COUNT them.
5. For "which/who" questions, copy the exact string from the table including any punctuation.
6. If the answer is not found in the table, output: N/A
7. Always end with: Final Answer: [your exact answer]

**EXAMPLE FORMAT FOR COUNTING:**
Question: How many songs were released after 2010?
Answer process:
- Row 1: "Song A" released 2012 ✓
- Row 2: "Song B" released 2008 ✗
- Row 3: "Song C" released 2015 ✓
Count: 2 songs
Final Answer: 2
"""
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
        "garbage_outputs_detected": 0,
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

        # Normalization
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

            # Run reasoning
            final_path, round_results, final_answer, decision_reason, stats = reasoning_with_parallel_analysis(
                base_model=model_obj,
                analyst_model=analyst_model,
                prompt=prompt,
                table=table,
                question=question,
                temperature=temperature,
                system=system,
                enable_thinking=enable_thinking
            )

            # Update global stats
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

            # Collect all answers
            all_answers = []
            for rr in round_results:
                for p in rr.paths:
                    all_answers.append(p.answer)

            # Save log
            log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
            with open(log_path, "w", encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write(f"3-Round + Parallel Analysis (FIXED) - Question #{global_i}\n")
                f.write("=" * 70 + "\n\n")
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
                        f.write(f"[Valid Format]: {path.is_valid_format}\n")  # [NEW]
                        f.write(f"[Reasoning]\n{path.reasoning}\n\n")
                        f.write(f"[Extracted Answer]: {path.answer}\n\n")
                        
                        status = "[OK] Correct" if analysis.is_correct else "[X] Error"
                        f.write(f"[Analyst Judgment]: {status} (Conf: {analysis.confidence_score:.2f})\n")  # [UPDATED]
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

            # Save result
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

    # Save statistics
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
