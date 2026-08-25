import os
import json

from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import Model

from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes
from utils.table import transpose, sort_dataframe

from run_helper import load_dataset, get_cot_prompt, query, check_transpose, check_sort, read_json_file


def main(
        # base model of the agent (for short prompt to save money)
        model: Optional[str] = None,
    # long model of the agent (only used for long prompt)
    long_model: Optional[str] = None,
        provider: str = "openai",  # openai, huggingface, vllm
        dataset: str = "train",  # wtq or tabfact
        perturbation: str = "none",  # none, transpose, shuffle, transpose_shuffle
        norm: bool = True,  # whether to NORM the table
        disable_resort: bool = True,  # whether to disable the resort stage in NORM
        # whether to cache the normalization results so that we can reuse them
        norm_cache: bool = True,
        sub_sample: bool = True,  # whether to only run on the subset sampled data points
        resume: int = 0,  # resume from the i-th data point
        stop_at: int = 1e6,  # stop at the i-th data point
    self_consistency: int = 10,  # how many times to do self consistency
    temperature: float = 0.1,  # temperature for model
    log_dir: str = "output/train/dp",  # directory to store the logs
    # directory to store the cache (normalization results)
    cache_dir: str = "cache",
    system: str = "You are a helpful assistant",  # system prompt for DeepSeek
    # whether to use strict format prompt for better Qwen output
    use_strict_format: bool = False,
    # whether to enable Qwen thinking mode (if supported by tokenizer)
    enable_thinking: bool = False,
):
    # 确保 enable_thinking 是布尔值（处理字符串输入）
    if isinstance(enable_thinking, str):
        enable_thinking = enable_thinking.lower() in ("true", "1", "yes", "on")
    enable_thinking = bool(enable_thinking)

    #### create log & cache dir and save config ####
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # store the config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w", encoding='utf-8') as f:
        json.dump({key: value for key, value in locals().items()
                  if key != 'f'}, f, indent=4)

    #### load dataset and cot prompt ####
    data = load_dataset(dataset)
    cot_prompt = get_cot_prompt(dataset, use_strict_format)

    #### load the model ####
    # 保存原始路径用于比较
    model_path = model
    long_model_path = long_model
    
    if model:
        model = Model(model, provider=provider)
    
    # 智能处理long_model：避免重复加载
    if long_model_path and model_path:
        # 如果long_model与model路径相同，复用同一实例以节省显存
        if long_model_path == model_path:
            print("⚠️  long_model与model路径相同，复用同一模型实例以节省显存")
            long_model = model
        else:
            print(f"⚠️  加载独立的long_model: {long_model_path}")
            long_model = Model(long_model_path, provider=provider)
    elif model:
        # 如果未指定long_model或为None，自动使用model作为long_model
        print("⚠️  未指定long_model，使用model作为long_model")
        long_model = model
    else:
        raise ValueError("必须至少指定一个模型（model或long_model）")

    #### load the cache ####
    transpose_cache = read_json_file(os.path.join(cache_dir, "transpose.json"))
    resort_cache = read_json_file(os.path.join(cache_dir, "resort.json"))

    #### prepare the iterator ####
    global_i = 0
    break_flag = False
    total = sum([len(d['sampled_indices']) for d in data]) if sub_sample else sum(
        [len(d['questions']) for d in data])
    # 修复进度条：显示实际要处理的数量 (stop_at - resume)
    actual_total = min(int(stop_at), total) - resume
    pbar = tqdm(total=max(actual_total, 0), desc=f"Processing {resume} to {min(int(stop_at), total)}")

    #### start the loop ####
    for table_idx, d in enumerate(data):
        if break_flag:
            break

        index_list = d['sampled_indices'] if sub_sample else range(
            len(d["questions"]))

        # if the table is empty, skip
        if len(index_list) == 0:
            continue

        # load table infos
        table_id = d["table_id"]
        title = d["title"]

        if perturbation == "none":  # 原始表格
            table = construct_markdown_table(**d["table"])
        elif perturbation == "transpose":  # 行列转置
            table = construct_markdown_table(**d["transposed_table"])
        elif perturbation == "shuffle":  # 行随机乱序
            table = construct_markdown_table(**d["row_shuffled_table"])
        elif perturbation == "transpose_shuffle":  # 先转置再乱序
            table = construct_markdown_table(
                **d["row_shuffled_transposed_table"])

        df = markdown_to_df(table)

        # transpose and sort if necessary
        transpose_flag = False
        resort_list = []

        if norm:
            transpose_flag = check_transpose(
                model, long_model, table, title, table_id, perturbation, transpose_cache, norm_cache, cache_dir)

            if transpose_flag:
                transposed_df = transpose(df)
                df = remove_merged_suffixes(transposed_df)

            if not disable_resort:
                resort_list = check_sort(
                    model, long_model, df, title, table_id, perturbation, resort_cache, norm_cache, cache_dir)
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

            text, response = query(
                model, long_model, prompt, temperature, self_consistency, system=system, enable_thinking=enable_thinking)

            log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding='utf-8') as f:
                f.write("===================Title===================\n")
                f.write(title + "\n")
                f.write("===================Table===================\n")
                f.write(table + "\n")
                f.write("===================Question===================\n")
                f.write(question + "\n")
                f.write("===================Text===================")
                if isinstance(text, str):
                    f.write(text)
                elif text is not None and hasattr(text, '__iter__') and not isinstance(text, (str, dict)):
                    f.write("\n".join(str(item) for item in text))
                else:
                    f.write(str(text) if text is not None else "None")
                f.write("\n")
                f.write("===================Answer===================\n")
                f.write(",".join(answer) if isinstance(
                    answer, list) else str(answer))
                f.write("\n")

            # No need for separate detailed reasoning log since it's in the main log

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
            }

            with open(os.path.join(log_dir, "result.jsonl"), "a", encoding='utf-8') as f:
                json.dump(res, f)
                f.write("\n")

            global_i += 1
            pbar.update(1)


if __name__ == "__main__":
    Fire(main)
