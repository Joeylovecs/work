import os
import json
from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import TableAgent, Model
from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes, convert_cells_to_numbers
from utils.table import transpose, sort_dataframe
from run_helper import load_dataset, check_transpose, check_sort, read_json_file


def main(
        model:Optional[str] = "DeepSeek-V3.1", # base model of the agent (for short prompt to save money)
        long_model:Optional[str] = "DeepSeek-V3.1", # long model of the agent (only used for long prompt)
        provider: str = "openai", # openai, huggingface, vllm
        dataset:str = "train", # wtq, tabfact
        perturbation: str = "none", # none, transpose, shuffle, transpose_shuffle
        use_full_table: bool = True, # whether to use the full table or only the partial table
        norm: bool = True, # whether to NORM the table
        disable_resort: bool = True, # whether to disable the resort stage in NORM
        norm_cache: bool = True, # whether to cache the normalization results so that we can reuse them
        sub_sample: bool = True, # whether to only run on the subset sampled data points
        resume:int = 0, # resume from the i-th data point
        stop_at:int = 2, # stop at the i-th data point
        self_consistency:int = 1, # how many times to do self consistency
        temperature:float=0.1, # temperature for model
        log_dir: str = "output/train/agent", # directory to store the logs
        cache_dir: str = "cache/DeepSeek-V3.1", # directory to store the cache (normalization results)
):
    
    #### create log & cache dir and save config ####
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # store the config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w", encoding='utf-8') as f:
        json.dump({key: value for key, value in locals().items() if key != 'f'}, f, indent=4)
    
    #### load dataset ####
    data = load_dataset(dataset)

    #### load the model ####
    if model:
        model = Model(model, provider=provider)
    if long_model:
        long_model = Model(long_model, provider=provider)
    
    #### load the cache ####
    transpose_cache = read_json_file(os.path.join(cache_dir, "transpose.json"))
    resort_cache = read_json_file(os.path.join(cache_dir, "resort.json"))
    
    #### prepare the iterator ####
    global_i = 0
    break_flag = False
    total = sum([len(d['sampled_indices']) for d in data]) if sub_sample else sum([len(d['questions']) for d in data])
    pbar = tqdm(total=stop_at if stop_at < total else total)

    # read the results from output/wtq_agent_wo_norm (if exists and has content)
    # Note: Set force_run_all=True to disable reusing and run all examples with NORM
    temp = []
    force_run_all = True  # Set to True to disable reusing previous results
    
    if not force_run_all:
        temp_file_path = "output/wtq_agent_wo_norm/result.jsonl"
        if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
            with open(temp_file_path, "r", encoding='utf-8') as f:
                temp = [json.loads(line) for line in f.readlines()]
            print(f"Loaded {len(temp)} results from {temp_file_path}")
        else:
            print(f"Warning: {temp_file_path} does not exist or is empty. Will run all examples.")
    else:
        print("Force running all examples with NORM processing. No reusing previous results.")
    
    #### start the loop ####
    for table_idx, d in enumerate(data):
        if break_flag:
            break

        index_list = d['sampled_indices'] if sub_sample else range(len(d["questions"]))
        
        # if the table is empty, skip
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
            transpose_flag = check_transpose(model, long_model, table, title, table_id, perturbation, transpose_cache, norm_cache, cache_dir)
            
            if transpose_flag:
                transposed_df = transpose(df)
                df = remove_merged_suffixes(transposed_df)
            
            if not disable_resort:
                resort_list = check_sort(model, long_model, df, title, table_id, perturbation, resort_cache, norm_cache, cache_dir)
                df = sort_dataframe(df, resort_list)
        
        df = convert_cells_to_numbers(df)
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

            # 只有当norm为True且存在有效的temp数据且不强制运行所有时才尝试复用
            if norm and not transpose_flag and temp and global_i < len(temp) and not force_run_all:
                # reuse the temp but also generate log file for annotation
                print(f"Reusing result {global_i} from temp and generating log", flush=True)
                
                temp_result = temp[global_i]
                
                # Generate log file for annotation purposes
                log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                
                with open(log_path, "w", encoding='utf-8') as f:
                    f.write("===================Title===================\n")
                    f.write(temp_result.get("title", "") + "\n")
                    f.write("===================Table===================\n")
                    f.write(temp_result.get("table", "") + "\n")
                    f.write("===================Question===================\n")
                    f.write(temp_result.get("question", "") + "\n")
                    f.write("===================Text===================\n")
                    text = temp_result.get("text", "")
                    f.write(text if isinstance(text, str) else "\n".join(text))
                    f.write("\n")
                    f.write("===================Answer===================\n")
                    answer = temp_result.get("answer", "")
                    f.write(",".join(answer) if isinstance(answer, list) else str(answer))
                    f.write("\n")
                
                # Write to result.jsonl
                with open(os.path.join(log_dir, "result.jsonl"), "a", encoding='utf-8') as f:
                    json.dump(temp_result, f)
                    f.write("\n")
                
                global_i += 1
                pbar.update(1)
                
                continue

            question = d["questions"][idx]
            answer = d["answers"][idx]
            question_id = d["ids"][idx]
            
            log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
            # create the file
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            texts = []
            
            for _ in range(self_consistency):  
                # create the table agent
                agent = TableAgent(
                    table=df,
                    prompt_type=dataset,
                    model=model,
                    long_model=long_model,
                    temperature=temperature,
                    log_dir=log_path,
                    use_full_table=use_full_table,
                )

                text, response = agent.run(question=question, title=title)
                texts.append(text)

            # Write detailed log file like in run_cot.py
            with open(log_path, "w", encoding='utf-8') as f:
                f.write("===================Title===================\n")
                f.write(title + "\n")
                f.write("===================Table===================\n")
                f.write(table + "\n")
                f.write("===================Question===================\n")
                f.write(question + "\n")
                f.write("===================Text===================\n")
                f.write(texts[0] if isinstance(texts[0], str) else "\n".join(texts[0]))
                f.write("\n")
                f.write("===================Answer===================\n")
                f.write(",".join(answer) if isinstance(answer, list) else str(answer))
                f.write("\n")


            res = {
                "idx": global_i,
                "answer": answer,
                "text": texts if self_consistency > 1 else texts[0],
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