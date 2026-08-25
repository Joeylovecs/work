import json
import os
import re


def extract_reasoning_and_final_answer(text, question):
    """
    从 text 字段中提取推理步骤和最终答案 (TabFact版本 - Llama)
    TabFact的特点: 
    - text中包含"Statement: ..."前缀
    - 答案是Yes/No而不是具体数值
    """
    # 1. 移除Statement前缀
    if text.startswith(f"Statement: {question}"):
        text = text[len(f"Statement: {question}"):].strip()
    elif text.startswith("Statement:"):
        text = re.sub(r"^Statement:.*?\n+", "", text, count=1)

    # 2. 提取最终答案
    final_answer = ""
    # 首先尝试匹配 "Final Answer:"
    final_answer_match = re.search(
        r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if final_answer_match:
        final_answer = final_answer_match.group(1).strip()
        # 从文本中移除 "Final Answer:" 及其之后的所有内容
        text = text[:final_answer_match.start()].strip()

    # 3. 将 "Therefore, the final answer is:" 或 "Therefore, based on..." 替换为简单的 "Therefore:" 标记
    text = re.sub(r"Therefore,\s+(?:the\s+)?(?:final\s+)?(?:answer\s+)?(?:is|based\s+on).*?:",
                  "Therefore:", text, flags=re.IGNORECASE)
    text = re.sub(r"Therefore,\s+based\s+on\s+the\s+table\s+data,\s+the\s+statement\s+is:\s+(?:True|False)",
                  "Therefore:", text, flags=re.IGNORECASE)

    # 4. 分割成独立的步骤
    individual_steps = re.split(r'(?=Step \d+:|Therefore:)', text)

    # 清理分割后可能产生的空字符串或仅包含空格的字符串
    individual_steps = [s.strip() for s in individual_steps if s.strip()]

    # 5. 生成累积步骤
    cumulative_steps = []
    if individual_steps:
        for i in range(1, len(individual_steps) + 1):
            cumulative_steps.append("\n".join(individual_steps[:i]))

    return cumulative_steps, final_answer


def standardize_sample(sample):
    """
    标准化单个 JSON 样本 (TabFact版本 - Llama)
    TabFact特点:
    - answer字段是整数0/1,不是列表
    - 需要保持原样,不转换
    """
    # 提取 is_correct 的值
    is_correct_val = sample.pop('is_correct', False)

    # 定义要移除的字段
    keys_to_remove = ['transpose', 'resort', 'question_id',
                      'title', 'pred_extracted', 'gold_joined', 'table_id', 'text']

    for key in keys_to_remove:
        sample.pop(key, None)

    # 提取处理所需的字段
    text_original = sample.pop('text_original_for_processing', '')
    if 'text' in sample:
        text_original = sample.pop('text')
    question = sample.get('question', '')

    # 获取标准化的推理步骤和最终答案
    reasoning_steps, final_answer = extract_reasoning_and_final_answer(
        text_original, question)

    # 按顺序构建新的样本字典
    # TabFact: answer是整数0/1,不是列表
    standardized_sample = {
        'idx': sample.get('idx'),
        'answer': sample.get('answer'),  # 保持原样(0或1)
        'table': sample.get('table'),
        'question': question,
        'reasoning_steps': reasoning_steps,
        'final_answer': final_answer,
        'is_correct': is_correct_val
    }
    
    # 移除所有值为 None 的键
    standardized_sample = {k: v for k,
                           v in standardized_sample.items() if v is not None}

    return standardized_sample


def process_file(input_path, output_path):
    """
    读取、处理并写入整个 JSONL 文件。
    """
    standardized_samples = []
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            if line.strip():
                try:
                    sample = json.loads(line)
                    sample['text_original_for_processing'] = sample.get(
                        'text', '')
                    standardized_sample = standardize_sample(sample)
                    standardized_samples.append(standardized_sample)
                except json.JSONDecodeError as e:
                    print(
                        f"Skipping line due to JSON decode error: {e} in file {input_path}")
                    continue

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for sample in standardized_samples:
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(
        f"Successfully standardized {len(standardized_samples)} samples from {input_path} to {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输入目录: output\llama_tabtrain
    input_dir = os.path.join(script_dir, '..', 'output', 'llama_tabtrain')
    
    # 输出目录: buzhou\training_data_zhengti_llama
    output_dir = os.path.join(script_dir, '..', 'buzhou', 'training_data_zhengti_llama')
    
    # 创建输出目录(如果不存在)
    os.makedirs(output_dir, exist_ok=True)

    positive_input = os.path.join(input_dir, 'positive_samples.jsonl')
    positive_output = os.path.join(output_dir, 'positive_samples_standardized.jsonl')

    negative_input = os.path.join(input_dir, 'negative_samples.jsonl')
    negative_output = os.path.join(output_dir, 'negative_samples_standardized.jsonl')

    print("Starting TabFact Llama training data standardization process...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    if os.path.exists(positive_input):
        process_file(positive_input, positive_output)
    else:
        print(f"Input file not found: {positive_input}")

    if os.path.exists(negative_input):
        process_file(negative_input, negative_output)
    else:
        print(f"Input file not found: {negative_input}")

    print("TabFact Llama training data standardization process finished.")
