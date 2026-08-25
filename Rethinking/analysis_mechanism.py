"""
==========================================================================
Mechanism Analysis Script for Primary Setting
==========================================================================

Primary Setting:
- Base Model: Qwen3-8B
- Thinking Mode: False
- Dataset: WTQ

分析目标：
  A. K=1,2,3 的性能趋势
  B. Early Stopping / Round Distribution
  C. Arbitration Source Analysis
  D. Efficiency / Token / Time Analysis

使用方法：
  python analysis_mechanism.py
  （在 Rethinking 目录下运行）

输出：
  - 控制台打印完整的 analysis facts audit 报告
  - 将结果写入 analysis_mechanism_results.json
==========================================================================
"""

import os
import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple

# ========================================================================
# 路径配置（相对于 Rethinking 目录）
# ========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 三组结果文件
PATHS = {
    # K=1 baseline: 单独 CoT（无多轮、无 analyst），qwen 非思考 跑 WTQ
    "dp": os.path.join(BASE_DIR, "output", "qwen_wtq", "qwen_fei_wtq_dp"),
    # K=3 full pipeline: qwen 非思考 + 微调后的 qwen 非思考 跑 WTQ
    "fenxishi": os.path.join(BASE_DIR, "fenxishi", "output", "final_5090_feisikao_wtq"),
    # K=3 full pipeline (ablation): qwen 非思考 + qwen 非思考 (无微调) 跑 WTQ
    "xiaorong": os.path.join(BASE_DIR, "xiaorong", "output", "xiao_final_feisikao_wtq"),
}


# ========================================================================
# 工具函数
# ========================================================================

def load_jsonl(filepath: str) -> List[dict]:
    """加载 JSONL 格式的结果文件"""
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def safe_div(a, b, default=0.0):
    return a / b if b > 0 else default


# ========================================================================
# Part A: K=1,2,3 的性能趋势
# ========================================================================

def compute_accuracy_at_K(results: List[dict], K: int) -> Tuple[float, int, int]:
    """
    从 full-pipeline (K=3) 的结果中，模拟截断到 K 轮时的准确率。
    
    逻辑：
    - 如果 early_exit_round <= K，使用该 early exit 时的 final_answer
    - 如果 early_exit_round > K 或无 early exit：
      - 只看前 K 轮的 paths，使用 decision_priority 逻辑重新选答案
      - 简化处理：从 all_answers 取前 K*3 个，做 majority vote
    
    注意：对于 K=1 的 dp baseline，直接从 dp 的 result.jsonl 计算
    """
    from utils.eval import eval_ex_match, extract_answer
    
    correct = 0
    total = 0
    
    for item in results:
        answer = ", ".join(item["answer"])
        total += 1
        
        # 如果该条记录有 early_exit_round 和 all_answers 字段（多轮 pipeline）
        if "early_exit_round" in item and "all_answers" in item:
            early_exit = item.get("early_exit_round")
            all_answers = item.get("all_answers", [])
            
            if early_exit is not None and early_exit <= K:
                # early exit 在 K 轮内发生，使用 final_answer
                pred = item.get("final_answer", "")
            else:
                # 截断到前 K*3 个 path 的答案，做 majority vote
                truncated = all_answers[:K * 3]
                # 提取答案
                preds = []
                for text in truncated:
                    if text and text != "N/A":
                        preds.append(text)
                if preds:
                    counter = Counter(preds)
                    pred = counter.most_common(1)[0][0]
                else:
                    pred = ""
            
            if eval_ex_match(answer, pred):
                correct += 1
        else:
            # dp baseline 格式：text 是单条文本
            text = item.get("text", "")
            if isinstance(text, list):
                text = text[0] if text else ""
            pred = extract_answer(text)
            if pred and eval_ex_match(answer, pred):
                correct += 1
    
    acc = safe_div(correct, total) * 100
    return acc, correct, total


def analyze_K_trend(dp_results: List[dict], pipeline_results: List[dict], 
                    pipeline_name: str) -> Dict[str, Any]:
    """分析 K=1,2,3 的性能趋势"""
    from utils.eval import eval_ex_match, extract_answer
    
    report = {
        "pipeline": pipeline_name,
        "K_results": {},
        "trend_analysis": {}
    }
    
    # K=1: 使用 dp baseline 的准确率
    dp_acc, dp_correct, dp_total = compute_accuracy_at_K(dp_results, K=1)
    report["K_results"]["K=1 (dp baseline)"] = {
        "accuracy": round(dp_acc, 2),
        "correct": dp_correct,
        "total": dp_total
    }
    
    # K=1 from pipeline: 模拟只用 Round 1 的结果
    k1_acc, k1_correct, k1_total = compute_accuracy_at_K(pipeline_results, K=1)
    report["K_results"]["K=1 (pipeline Round1 only)"] = {
        "accuracy": round(k1_acc, 2),
        "correct": k1_correct,
        "total": k1_total
    }
    
    # K=2
    k2_acc, k2_correct, k2_total = compute_accuracy_at_K(pipeline_results, K=2)
    report["K_results"]["K=2"] = {
        "accuracy": round(k2_acc, 2),
        "correct": k2_correct,
        "total": k2_total
    }
    
    # K=3 (原始结果)
    k3_correct = 0
    k3_total = 0
    for item in pipeline_results:
        answer = ", ".join(item["answer"])
        pred = item.get("final_answer", "")
        k3_total += 1
        if pred and eval_ex_match(answer, pred):
            k3_correct += 1
    k3_acc = safe_div(k3_correct, k3_total) * 100
    report["K_results"]["K=3 (full pipeline)"] = {
        "accuracy": round(k3_acc, 2),
        "correct": k3_correct,
        "total": k3_total
    }
    
    # 趋势分析
    gain_1_to_2 = k2_acc - k1_acc
    gain_2_to_3 = k3_acc - k2_acc
    report["trend_analysis"] = {
        "K1_pipeline_acc": round(k1_acc, 2),
        "K2_acc": round(k2_acc, 2),
        "K3_acc": round(k3_acc, 2),
        "gain_K1_to_K2": round(gain_1_to_2, 2),
        "gain_K2_to_K3": round(gain_2_to_3, 2),
        "marginal_gain_decreasing": gain_2_to_3 < gain_1_to_2,
        "K3_still_improves_over_K2": gain_2_to_3 > 0
    }
    
    return report


# ========================================================================
# Part B: Early Stopping / Round Distribution
# ========================================================================

def analyze_early_stopping(results: List[dict], pipeline_name: str) -> Dict[str, Any]:
    """统计 early stopping 相关指标"""
    
    total = len(results)
    
    # stop@1, stop@2, stop@3 的计数
    stop_at = {1: 0, 2: 0, 3: 0}
    
    total_rounds_sum = 0
    total_paths_sum = 0
    
    for item in results:
        early_exit = item.get("early_exit_round")
        total_rounds_val = item.get("total_rounds", 3)
        total_paths_val = item.get("total_paths", 9)
        
        if early_exit is not None:
            if early_exit in stop_at:
                stop_at[early_exit] += 1
            actual_rounds = early_exit
        else:
            # 没有 early exit => 跑满 3 轮
            stop_at[3] += 1
            actual_rounds = 3
        
        total_rounds_sum += actual_rounds
        total_paths_sum += total_paths_val
    
    avg_rounds = safe_div(total_rounds_sum, total)
    avg_paths = safe_div(total_paths_sum, total)
    
    report = {
        "pipeline": pipeline_name,
        "total_examples": total,
        "stop_distribution": {
            "stop@1": stop_at[1],
            "stop@2": stop_at[2],
            "stop@3": stop_at[3],
            "stop@1_pct": round(safe_div(stop_at[1], total) * 100, 2),
            "stop@2_pct": round(safe_div(stop_at[2], total) * 100, 2),
            "stop@3_pct": round(safe_div(stop_at[3], total) * 100, 2),
        },
        "average_executed_rounds": round(avg_rounds, 4),
        "average_total_paths": round(avg_paths, 4),
        "full_budget_paths": 9,  # K=3, 每轮 3 paths
        "savings_vs_full_budget_pct": round((1 - safe_div(avg_paths, 9)) * 100, 2)
    }
    
    return report


# ========================================================================
# Part C: Arbitration Source Analysis
# ========================================================================

def analyze_arbitration_source(results: List[dict], pipeline_name: str) -> Dict[str, Any]:
    """
    统计最终答案的来源：
    
    映射规则：
    1. final_round_verified:
       - decision_priority 包含 "Round3_correct" 或 "Round3_unanimous"
       - 即 Priority1 或 Round3 的 unanimous
    
    2. historical_verified:
       - decision_priority 包含 "Priority2" 或 "Round1_unanimous" 或 "Round2_unanimous"
       - 即 Arbiter 回溯到历史轮次中验证正确的候选
    
    3. global_fallback:
       - decision_priority 包含 "Priority3"
       - 即所有路径都被判为错误，退化为全局投票
    """
    
    total = len(results)
    
    source_counts = {
        "final_round_verified": 0,       # Priority1 (Round3 correct)
        "historical_verified": 0,         # Priority2 (Round1-2 correct) + Round1/2 unanimous
        "global_fallback": 0,             # Priority3 (all wrong, vote)
    }
    
    # 更细粒度的分类
    detailed_counts = defaultdict(int)
    
    for item in results:
        dp = item.get("decision_priority", "")
        
        detailed_counts[dp] += 1
        
        if dp is None:
            dp = ""
        
        # 映射到三类
        if "Round3_unanimous" in dp:
            # Round3 全部一致 -> final_round_verified
            source_counts["final_round_verified"] += 1
        elif "Priority1" in dp and "Round3" in dp:
            # Priority1: Round3 有 correct path
            source_counts["final_round_verified"] += 1
        elif "Round1_unanimous" in dp or "Round2_unanimous" in dp:
            # Round1/2 全部一致 -> early exit -> historical_verified
            source_counts["historical_verified"] += 1
        elif "Priority2" in dp:
            # Priority2: 回溯到 Round1-2 的 correct paths
            source_counts["historical_verified"] += 1
        elif "Priority3" in dp:
            # Priority3: 全部 wrong -> global fallback vote
            source_counts["global_fallback"] += 1
        else:
            # 未知类型 -> 归入 global_fallback
            source_counts["global_fallback"] += 1
    
    report = {
        "pipeline": pipeline_name,
        "total_examples": total,
        "arbitration_sources": {
            k: {
                "count": v,
                "pct": round(safe_div(v, total) * 100, 2)
            }
            for k, v in source_counts.items()
        },
        "detailed_decision_priorities": {
            k: v for k, v in sorted(detailed_counts.items(), key=lambda x: -x[1])
        },
        "hierarchical_arbitration_active": (
            source_counts["historical_verified"] > 0 or 
            source_counts["final_round_verified"] > 0
        ),
        "hierarchical_usage_pct": round(
            safe_div(source_counts["final_round_verified"] + source_counts["historical_verified"], total) * 100, 2
        )
    }
    
    return report


# ========================================================================
# Part D: Efficiency / Token / Time Analysis
# ========================================================================

def analyze_efficiency(results: List[dict], pipeline_name: str) -> Dict[str, Any]:
    """
    分析 token 和时间消耗。
    
    注意：result.jsonl 和 log 文件中没有直接的 token/time 记录。
    因此：
    - token 消耗：MISSING（日志中无 token 计数）
    - 时间消耗：MISSING（日志中无时间记录）
    
    但我们可以构造 full-budget upper bound 对照：
    - actual avg paths = 从 B 计算
    - full-budget paths = 9 (K=3, 每轮 3)
    - savings ratio = 1 - actual/full
    """
    
    total = len(results)
    
    # 检查是否有 token 相关字段
    has_token_info = False
    has_time_info = False
    token_fields = ["total_tokens", "prompt_tokens", "completion_tokens", "token_count"]
    time_fields = ["generation_time", "time_cost", "elapsed", "duration"]
    
    sample = results[0] if results else {}
    for field in token_fields:
        if field in sample:
            has_token_info = True
            break
    for field in time_fields:
        if field in sample:
            has_time_info = True
            break
    
    # 计算 path-based cost proxy
    total_paths_sum = 0
    full_budget_paths = 9 * total  # 每个样本 9 paths
    
    for item in results:
        total_paths_val = item.get("total_paths", 9)
        total_paths_sum += total_paths_val
    
    actual_avg_paths = safe_div(total_paths_sum, total)
    
    report = {
        "pipeline": pipeline_name,
        "total_examples": total,
        "token_analysis": {
            "status": "AVAILABLE" if has_token_info else "MISSING",
            "reason": "result.jsonl 中没有 token 统计字段" if not has_token_info else "从结果文件中提取"
        },
        "time_analysis": {
            "status": "AVAILABLE" if has_time_info else "MISSING",
            "reason": "result.jsonl 和 log 文件中没有时间记录字段" if not has_time_info else "从结果文件中提取"
        },
        "path_based_cost_proxy": {
            "actual_total_paths": total_paths_sum,
            "actual_avg_paths_per_example": round(actual_avg_paths, 4),
            "full_budget_paths_per_example": 9,
            "full_budget_total_paths": full_budget_paths,
            "savings_ratio_pct": round((1 - safe_div(total_paths_sum, full_budget_paths)) * 100, 2),
            "note": "path 数量是推理成本的可靠 proxy：每条 path = 1 次 base model generation + 1 次 analyst evaluation"
        }
    }
    
    # 如果有 token 信息，提取具体数值
    if has_token_info:
        total_tokens = sum(item.get("total_tokens", 0) for item in results)
        report["token_analysis"]["total_tokens"] = total_tokens
        report["token_analysis"]["avg_tokens_per_example"] = round(safe_div(total_tokens, total), 2)
    
    if has_time_info:
        total_time = sum(item.get("generation_time", item.get("time_cost", item.get("elapsed", 0))) for item in results)
        report["time_analysis"]["total_time_sec"] = round(total_time, 2)
        report["time_analysis"]["avg_time_per_example_sec"] = round(safe_div(total_time, total), 2)
    
    return report


# ========================================================================
# Part E: 综合报告生成
# ========================================================================

def generate_full_report():
    """生成完整的 analysis facts audit 报告"""
    
    print("=" * 80)
    print("  MECHANISM ANALYSIS - Primary Setting Facts Audit")
    print("  Qwen3-8B / Thinking=False / WTQ")
    print("=" * 80)
    
    # 加载数据
    print("\n[1/6] 加载结果文件...")
    
    # 将 utils 路径加入 sys.path
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    
    dp_result_path = os.path.join(PATHS["dp"], "result.jsonl")
    fenxishi_result_path = os.path.join(PATHS["fenxishi"], "result.jsonl")
    xiaorong_result_path = os.path.join(PATHS["xiaorong"], "result.jsonl")
    
    dp_results = load_jsonl(dp_result_path)
    fenxishi_results = load_jsonl(fenxishi_result_path)
    xiaorong_results = load_jsonl(xiaorong_result_path)
    
    print(f"  dp (K=1 baseline): {len(dp_results)} examples")
    print(f"  fenxishi (finetuned analyst): {len(fenxishi_results)} examples")
    print(f"  xiaorong (non-finetuned analyst): {len(xiaorong_results)} examples (部分完成)")
    
    all_reports = {}
    
    # ====================================================================
    # 第一部分：可用日志与脚本概览
    # ====================================================================
    print("\n" + "=" * 80)
    print("  第一部分：Primary Setting 的可用日志与脚本概览")
    print("=" * 80)
    
    file_inventory = {
        "dp_baseline (K=1, 无 analyst)": {
            "result.jsonl": dp_result_path,
            "config.json": os.path.join(PATHS["dp"], "config.json"),
            "log_dir": os.path.join(PATHS["dp"], "log"),
            "log_count": len(os.listdir(os.path.join(PATHS["dp"], "log"))),
            "result_count": len(dp_results),
            "key_fields": "idx, answer, text, question_id (无 early_exit_round, 无 decision_priority)",
            "evidence": "result.jsonl 中每条仅含单条 text 字段 -> 标准 CoT"
        },
        "fenxishi (K=3, finetuned analyst)": {
            "result.jsonl": fenxishi_result_path,
            "config.json": os.path.join(PATHS["fenxishi"], "config.json"),
            "statistics.json": os.path.join(PATHS["fenxishi"], "statistics.json"),
            "log_dir": os.path.join(PATHS["fenxishi"], "log"),
            "log_count": len(os.listdir(os.path.join(PATHS["fenxishi"], "log"))),
            "result_count": len(fenxishi_results),
            "key_fields": "total_rounds, total_paths, correct_paths, all_answers, final_answer, "
                          "selected_path_id, decision_reason, decision_priority, early_exit_round",
            "evidence": "result.jsonl 每条含 early_exit_round 和 decision_priority 字段"
        },
        "xiaorong (K=3, non-finetuned analyst)": {
            "result.jsonl": xiaorong_result_path,
            "config.json": os.path.join(PATHS["xiaorong"], "config.json"),
            "log_dir": os.path.join(PATHS["xiaorong"], "log"),
            "log_count": len(os.listdir(os.path.join(PATHS["xiaorong"], "log"))),
            "result_count": len(xiaorong_results),
            "key_fields": "total_rounds, total_paths, correct_paths, all_answers, final_answer, "
                          "selected_path_id, decision_reason, decision_priority, early_exit_round, "
                          "enable_thinking, enable_thinking_analyst",
            "evidence": "result.jsonl 每条含 enable_thinking=false, enable_thinking_analyst=false"
        }
    }
    
    for name, info in file_inventory.items():
        print(f"\n  [{name}]")
        for k, v in info.items():
            print(f"    {k}: {v}")
    
    print("\n  推理主脚本：")
    print(f"    fenxishi: {os.path.join(BASE_DIR, 'fenxishi', 'run_cot_with_voting_fixed.py')}")
    print(f"      - early stopping: 函数 check_unanimous_correct() + reasoning_with_parallel_analysis() L955")
    print(f"      - final decision: 函数 final_decision() L790-L852, 3-priority 逻辑")
    print(f"    xiaorong: {os.path.join(BASE_DIR, 'xiaorong', 'xiaoHPCrun_cot_sikao.py')}")
    print(f"      - 同一套 3-priority 决策逻辑")
    print(f"    evaluate: {os.path.join(BASE_DIR, 'evaluate.py')}")
    print(f"      - 准确率计算: eval_wtq() with majority voting")
    
    all_reports["file_inventory"] = file_inventory
    
    # ====================================================================
    # 第二部分：K=1,2,3 的性能趋势
    # ====================================================================
    print("\n" + "=" * 80)
    print("  第二部分：K=1,2,3 的性能趋势")
    print("=" * 80)
    
    # 对 fenxishi（主要 pipeline）进行分析
    print("\n  [fenxishi pipeline - finetuned analyst]")
    k_trend_fenxishi = analyze_K_trend(dp_results, fenxishi_results, "fenxishi")
    
    for k_name, k_info in k_trend_fenxishi["K_results"].items():
        print(f"    {k_name}: {k_info['accuracy']}% ({k_info['correct']}/{k_info['total']})")
    
    trend = k_trend_fenxishi["trend_analysis"]
    print(f"\n    Gain K=1 → K=2: +{trend['gain_K1_to_K2']}%")
    print(f"    Gain K=2 → K=3: +{trend['gain_K2_to_K3']}%")
    print(f"    边际增益递减: {trend['marginal_gain_decreasing']}")
    print(f"    K=3 仍优于 K=2: {trend['K3_still_improves_over_K2']}")
    
    # 对 xiaorong（ablation pipeline）进行分析
    print("\n  [xiaorong pipeline - non-finetuned analyst] (注意: 仅 3601 samples)")
    k_trend_xiaorong = analyze_K_trend(dp_results, xiaorong_results, "xiaorong")
    
    for k_name, k_info in k_trend_xiaorong["K_results"].items():
        print(f"    {k_name}: {k_info['accuracy']}% ({k_info['correct']}/{k_info['total']})")
    
    trend_xr = k_trend_xiaorong["trend_analysis"]
    print(f"\n    Gain K=1 → K=2: +{trend_xr['gain_K1_to_K2']}%")
    print(f"    Gain K=2 → K=3: +{trend_xr['gain_K2_to_K3']}%")
    print(f"    边际增益递减: {trend_xr['marginal_gain_decreasing']}")
    print(f"    K=3 仍优于 K=2: {trend_xr['K3_still_improves_over_K2']}")
    
    all_reports["K_trend"] = {
        "fenxishi": k_trend_fenxishi,
        "xiaorong": k_trend_xiaorong
    }
    
    # ====================================================================
    # 第三部分：Early Stopping / Round Distribution
    # ====================================================================
    print("\n" + "=" * 80)
    print("  第三部分：Early Stopping / Round Distribution")
    print("=" * 80)
    
    print("\n  [fenxishi pipeline]")
    es_fenxishi = analyze_early_stopping(fenxishi_results, "fenxishi")
    sd = es_fenxishi["stop_distribution"]
    print(f"    Total examples: {es_fenxishi['total_examples']}")
    print(f"    stop@1: {sd['stop@1']} ({sd['stop@1_pct']}%)")
    print(f"    stop@2: {sd['stop@2']} ({sd['stop@2_pct']}%)")
    print(f"    stop@3: {sd['stop@3']} ({sd['stop@3_pct']}%)")
    print(f"    Average executed rounds: {es_fenxishi['average_executed_rounds']}")
    print(f"    Average total paths: {es_fenxishi['average_total_paths']}")
    print(f"    Savings vs full budget: {es_fenxishi['savings_vs_full_budget_pct']}%")
    
    print("\n  [xiaorong pipeline]")
    es_xiaorong = analyze_early_stopping(xiaorong_results, "xiaorong")
    sd_xr = es_xiaorong["stop_distribution"]
    print(f"    Total examples: {es_xiaorong['total_examples']}")
    print(f"    stop@1: {sd_xr['stop@1']} ({sd_xr['stop@1_pct']}%)")
    print(f"    stop@2: {sd_xr['stop@2']} ({sd_xr['stop@2_pct']}%)")
    print(f"    stop@3: {sd_xr['stop@3']} ({sd_xr['stop@3_pct']}%)")
    print(f"    Average executed rounds: {es_xiaorong['average_executed_rounds']}")
    print(f"    Average total paths: {es_xiaorong['average_total_paths']}")
    print(f"    Savings vs full budget: {es_xiaorong['savings_vs_full_budget_pct']}%")
    
    all_reports["early_stopping"] = {
        "fenxishi": es_fenxishi,
        "xiaorong": es_xiaorong
    }
    
    # ====================================================================
    # 第四部分：Arbitration Source Analysis
    # ====================================================================
    print("\n" + "=" * 80)
    print("  第四部分：Arbitration Source Analysis")
    print("=" * 80)
    
    print("\n  [fenxishi pipeline]")
    arb_fenxishi = analyze_arbitration_source(fenxishi_results, "fenxishi")
    for src, info in arb_fenxishi["arbitration_sources"].items():
        print(f"    {src}: {info['count']} ({info['pct']}%)")
    print(f"    Hierarchical arbitration active: {arb_fenxishi['hierarchical_arbitration_active']}")
    print(f"    Hierarchical usage: {arb_fenxishi['hierarchical_usage_pct']}%")
    print(f"    Detailed priorities:")
    for dp_name, dp_count in arb_fenxishi["detailed_decision_priorities"].items():
        print(f"      {dp_name}: {dp_count}")
    
    print("\n  [xiaorong pipeline]")
    arb_xiaorong = analyze_arbitration_source(xiaorong_results, "xiaorong")
    for src, info in arb_xiaorong["arbitration_sources"].items():
        print(f"    {src}: {info['count']} ({info['pct']}%)")
    print(f"    Hierarchical arbitration active: {arb_xiaorong['hierarchical_arbitration_active']}")
    print(f"    Hierarchical usage: {arb_xiaorong['hierarchical_usage_pct']}%")
    print(f"    Detailed priorities:")
    for dp_name, dp_count in arb_xiaorong["detailed_decision_priorities"].items():
        print(f"      {dp_name}: {dp_count}")
    
    all_reports["arbitration"] = {
        "fenxishi": arb_fenxishi,
        "xiaorong": arb_xiaorong
    }
    
    # ====================================================================
    # 第五部分：Efficiency / Token / Time Analysis
    # ====================================================================
    print("\n" + "=" * 80)
    print("  第五部分：Efficiency / Token / Time Analysis")
    print("=" * 80)
    
    print("\n  [fenxishi pipeline]")
    eff_fenxishi = analyze_efficiency(fenxishi_results, "fenxishi")
    print(f"    Token 分析: {eff_fenxishi['token_analysis']['status']}")
    print(f"      原因: {eff_fenxishi['token_analysis']['reason']}")
    print(f"    Time 分析: {eff_fenxishi['time_analysis']['status']}")
    print(f"      原因: {eff_fenxishi['time_analysis']['reason']}")
    pc = eff_fenxishi["path_based_cost_proxy"]
    print(f"    Path-based cost proxy:")
    print(f"      Actual avg paths/example: {pc['actual_avg_paths_per_example']}")
    print(f"      Full budget paths/example: {pc['full_budget_paths_per_example']}")
    print(f"      Savings ratio: {pc['savings_ratio_pct']}%")
    
    print("\n  [xiaorong pipeline]")
    eff_xiaorong = analyze_efficiency(xiaorong_results, "xiaorong")
    print(f"    Token 分析: {eff_xiaorong['token_analysis']['status']}")
    print(f"    Time 分析: {eff_xiaorong['time_analysis']['status']}")
    pc_xr = eff_xiaorong["path_based_cost_proxy"]
    print(f"    Path-based cost proxy:")
    print(f"      Actual avg paths/example: {pc_xr['actual_avg_paths_per_example']}")
    print(f"      Full budget paths/example: {pc_xr['full_budget_paths_per_example']}")
    print(f"      Savings ratio: {pc_xr['savings_ratio_pct']}%")
    
    all_reports["efficiency"] = {
        "fenxishi": eff_fenxishi,
        "xiaorong": eff_xiaorong
    }
    
    # ====================================================================
    # 第六部分：主文 vs Appendix 推荐
    # ====================================================================
    print("\n" + "=" * 80)
    print("  第六部分：主文 vs Appendix 推荐")
    print("=" * 80)
    
    recommendations = {
        "主文 Analysis 推荐": [
            {
                "内容": "K=1,2,3 accuracy trend (fenxishi pipeline)",
                "原因": "直接支撑 K=3 的 empirical justification，核心结论",
                "数据来源": "Part A"
            },
            {
                "内容": "Early Stopping distribution (stop@1/2/3 比例 + average rounds)",
                "原因": "直观展示 early stopping 的效果，节省多少计算量",
                "数据来源": "Part B"
            },
            {
                "内容": "Arbitration Source 三分类比例 (final_round / historical / fallback)",
                "原因": "证明 hierarchical arbitration 在发挥作用，不是全部 fallback",
                "数据来源": "Part C"
            }
        ],
        "Appendix 推荐": [
            {
                "内容": "xiaorong (ablation) 的对比数据",
                "原因": "ablation 对比用于验证 finetuned analyst 的效果差异，不是主要结论",
            },
            {
                "内容": "Detailed decision_priority 分布",
                "原因": "过于技术细节，主文表/图中用三分类汇总即可",
            },
            {
                "内容": "dp baseline 的详细准确率对比",
                "原因": "dp 是 CoT baseline，已在主表中报告过",
            },
            {
                "内容": "Token/Time 的 full-budget upper bound 构造（如未来补充 token 数据）",
                "原因": "目前 token/time 数据 MISSING，待补充后可放 appendix",
            }
        ]
    }
    
    for section, items in recommendations.items():
        print(f"\n  [{section}]")
        for i, item in enumerate(items, 1):
            print(f"    {i}. {item['内容']}")
            print(f"       原因: {item['原因']}")
    
    all_reports["recommendations"] = recommendations
    
    # ====================================================================
    # 第七部分：下一轮写 Analysis 正文必须补充的信息清单
    # ====================================================================
    print("\n" + "=" * 80)
    print("  第七部分：写 Analysis 正文前必须补充的最小信息清单")
    print("=" * 80)
    
    questions = [
        "1. xiaorong 的剩余 ~744 条数据跑完后，是否需要重新运行本脚本？(预期：是)",
        "2. 是否有独立的 K=1 和 K=2 的 pipeline 运行结果？当前 K=1/K=2 是从 K=3 结果中截断模拟的，若有独立运行会更严谨。",
        "3. 论文中 K=3 的 full pipeline accuracy 与你之前报告的最终数字是否一致？请确认 fenxishi 的官方准确率。",
        "4. 推理脚本中每条 path 的 analyst 调用次数是否固定为 1 次（或 n_votes=3 次）？这影响 cost proxy 的倍率计算。",
        "5. 是否需要报告 per-round correct path 比例的递增趋势（即 error correction 效果）？如果是，我可以从 all_answers + analyst judgment 中抽取。",
        "6. Token 和 Time 数据：是否可以通过 vLLM/HuggingFace 的日志文件（如 GPU profiling log）补充？如有请提供路径。",
        "7. 你论文中对 arbitration source 的三分类命名是否与我使用的 (final_round_verified / historical_verified / global_fallback) 一致？",
        "8. 主文 Analysis 的篇幅预期是多少？(如 1 page / 0.5 page) 以便决定详略程度。"
    ]
    
    for q in questions:
        print(f"  {q}")
    
    all_reports["pending_questions"] = questions
    
    # ====================================================================
    # 保存结果到 JSON
    # ====================================================================
    output_path = os.path.join(BASE_DIR, "analysis_mechanism_results.json")
    
    # 清理不可序列化的内容
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (bool,)):
            return obj
        elif isinstance(obj, (int, float, str)):
            return obj
        elif obj is None:
            return None
        else:
            return str(obj)
    
    serializable_reports = make_serializable(all_reports)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_reports, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"  完整报告已保存至: {output_path}")
    print(f"{'=' * 80}")
    
    return all_reports


# ========================================================================
# 入口
# ========================================================================
if __name__ == "__main__":
    generate_full_report()
