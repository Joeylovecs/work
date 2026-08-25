#!/usr/bin/env python3
"""
构建正向和负向训练集的脚本

基于DeepSeek正确答案和Qwen错误答案的交集来构建训练数据：
- 正向训练集：DeepSeek答对且Qwen答错的样本，使用DeepSeek的正确答案
- 负向训练集：DeepSeek答对且Qwen答错的样本，使用Qwen的错误答案
"""

import json
import argparse
import os
from typing import Dict, Set, Tuple
import sys

# 确保项目根目录在sys.path中，以便导入utils模块
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_jsonl(file_path: str) -> Dict[str, dict]:
    """
    加载JSONL文件并返回以question_id为键的字典
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # 使用question_id作为键，如果没有则使用idx
                key = item.get('question_id', item.get('idx', str(len(data))))
                data[key] = item
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {line[:100]}...")
                continue
    return data


def load_json(file_path: str) -> Dict[str, dict]:
    """
    加载JSON文件并返回以question_id为键的字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    data = {}
    for item in data_list:
        # 使用question_id作为键，如果没有则使用idx
        key = item.get('question_id', item.get('idx', str(len(data))))
        data[key] = item
    return data


def find_intersection_samples(deepseek_correct: Dict[str, dict],
                              qwen_incorrect: Dict[str, dict]) -> Tuple[Set[str], Dict[str, dict], Dict[str, dict]]:
    """
    找到DeepSeek答对且Qwen答错的样本交集

    Returns:
        intersection_ids: 交集的question_id集合
        positive_samples: 正向训练样本 (DeepSeek的正确答案)
        negative_samples: 负向训练样本 (Qwen的错误答案)
    """
    deepseek_ids = set(deepseek_correct.keys())
    qwen_ids = set(qwen_incorrect.keys())

    # 找到交集
    intersection_ids = deepseek_ids & qwen_ids

    print(f"DeepSeek正确答案数量: {len(deepseek_ids)}")
    print(f"Qwen错误答案数量: {len(qwen_ids)}")
    print(f"交集数量 (DeepSeek答对且Qwen答错): {len(intersection_ids)}")

    # 构建正向和负向训练样本
    positive_samples = {}
    negative_samples = {}

    for qid in intersection_ids:
        # 正向样本：使用DeepSeek的正确答案（保持原样，不添加元字段）
        positive_samples[qid] = deepseek_correct[qid].copy()

        # 负向样本：使用Qwen的错误答案（保持原样，不添加元字段）
        negative_samples[qid] = qwen_incorrect[qid].copy()

    return intersection_ids, positive_samples, negative_samples


def save_training_data(samples: Dict[str, dict], output_path: str, description: str):
    """
    保存训练数据到JSONL文件
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Sort samples by numeric idx if present, otherwise by question id

    def _get_idx_for_sort(item: dict):
        for k in ('idx', 'id', 'index'):
            if k in item:
                try:
                    return int(item[k])
                except Exception:
                    try:
                        return int(str(item[k]))
                    except Exception:
                        return None
        return None

    items = list(samples.items())
    if any(_get_idx_for_sort(s) is not None for _, s in items):
        items.sort(key=lambda kv: (_get_idx_for_sort(
            kv[1]) if _get_idx_for_sort(kv[1]) is not None else float('inf')))
    else:
        items.sort(key=lambda kv: kv[0])

    with open(output_path, 'w', encoding='utf-8') as f:
        for qid, sample in items:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"{description}已保存: {output_path} ({len(samples)} 样本)")


def main():
    parser = argparse.ArgumentParser(description="构建基于DeepSeek和Qwen结果对比的训练集")
    parser.add_argument("--deepseek_correct",
                        default="output/train/dp/correct.jsonl",
                        help="DeepSeek正确答案文件路径 (默认: output/train/dp/correct.jsonl)")
    parser.add_argument("--qwen_incorrect",
                        default="incorrect.jsonl",
                        help="Qwen错误答案文件路径 (默认: incorrect.jsonl)")
    parser.add_argument("--output_dir",
                        default="training_data",
                        help="输出目录 (默认: training_data)")
    parser.add_argument("--positive_name",
                        default="positive_samples.jsonl",
                        help="正向训练集文件名 (默认: positive_samples.jsonl)")
    parser.add_argument("--negative_name",
                        default="negative_samples.jsonl",
                        help="负向训练集文件名 (默认: negative_samples.jsonl)")

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.deepseek_correct):
        print(f"错误: DeepSeek正确答案文件不存在: {args.deepseek_correct}")
        return

    if not os.path.exists(args.qwen_incorrect):
        print(f"错误: Qwen错误答案文件不存在: {args.qwen_incorrect}")
        return

    print("🚀 开始构建训练集...")
    print(f"DeepSeek正确答案文件: {args.deepseek_correct}")
    print(f"Qwen错误答案文件: {args.qwen_incorrect}")

    # 加载数据
    print("\n📂 加载数据文件...")
    try:
        # 根据文件扩展名选择加载方式
        if args.deepseek_correct.endswith('.json'):
            deepseek_correct = load_json(args.deepseek_correct)
        else:
            deepseek_correct = load_jsonl(args.deepseek_correct)

        qwen_incorrect = load_jsonl(args.qwen_incorrect)

    except Exception as e:
        print(f"错误: 加载文件失败: {e}")
        return

    # 找到交集并构建训练样本
    print("\n🔍 分析数据交集...")
    intersection_ids, positive_samples, negative_samples = find_intersection_samples(
        deepseek_correct, qwen_incorrect
    )

    if not intersection_ids:
        print("⚠️  警告: 没有找到交集样本！")
        return

    # 保存训练数据
    print("\n💾 保存训练数据...")
    positive_path = os.path.join(args.output_dir, args.positive_name)
    negative_path = os.path.join(args.output_dir, args.negative_name)

    save_training_data(positive_samples, positive_path, "正向训练集")
    save_training_data(negative_samples, negative_path, "负向训练集")

    # 保存统计信息
    stats = {
        "deepseek_correct_count": len(deepseek_correct),
        "qwen_incorrect_count": len(qwen_incorrect),
        "intersection_count": len(intersection_ids),
        "positive_samples_count": len(positive_samples),
        "negative_samples_count": len(negative_samples),
        "intersection_ratio": len(intersection_ids) / max(len(deepseek_correct), len(qwen_incorrect)),
        "files": {
            "deepseek_correct": args.deepseek_correct,
            "qwen_incorrect": args.qwen_incorrect,
            "positive_samples": positive_path,
            "negative_samples": negative_path
        }
    }

    stats_path = os.path.join(args.output_dir, "training_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"统计信息已保存: {stats_path}")

    # 打印统计摘要
    print("\n📊 构建完成统计:")
    print(f"  - DeepSeek正确答案: {stats['deepseek_correct_count']} 个")
    print(f"  - Qwen错误答案: {stats['qwen_incorrect_count']} 个")
    print(f"  - 交集样本: {stats['intersection_count']} 个")
    print(f"  - 交集比例: {stats['intersection_ratio']:.2%}")
    print(f"  - 正向训练集: {stats['positive_samples_count']} 个样本")
    print(f"  - 负向训练集: {stats['negative_samples_count']} 个样本")

    print("\n✅ 训练集构建完成！")


if __name__ == "__main__":
    main()
