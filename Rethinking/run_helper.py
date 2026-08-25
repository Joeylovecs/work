import os
import json
from agent.model import Model
import re
from utils.data import print_partial_markdown
from utils.eval import parse_header_checking_result, parse_header_sorting_result


def load_dataset(dataset_name=None, dataset_file=None):
    """
    Load the dataset based on the dataset name, either from dataset name or dataset file.

    Args:
    - dataset_name (str): The name of the dataset.
    - dataset_file (str): The path to the dataset file.

    Returns:
    - dict: The dataset.
    """
    if dataset_name in ["wtq", "wikitablequestion"]:
        with open("data/wtq.json", "r", encoding='utf-8') as f:
            data = json.load(f)
    elif dataset_name in ["tabfact", "tabularfact"]:
        with open("data/tabfact.json", "r", encoding='utf-8') as f:
            data = json.load(f)
    elif dataset_name == "train":
        with open("data/train.json", "r", encoding='utf-8') as f:
            data = json.load(f)
    elif dataset_name == "valid":
        with open("data/valid.json", "r", encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Load the dataset from the file
        if dataset_file is None:
            raise ValueError(
                f"Dataset {dataset_name} is not supported, please provide a dataset file.")
        with open(dataset_file, "r", encoding='utf-8') as f:
            data = json.load(f)
    return data


def get_cot_prompt(dataset_name, use_strict_format=False):
    """
    Load the COT prompt based on the dataset name.

    Args:
    - dataset_name (str): The name of the dataset.
    - use_strict_format (bool): Whether to use strict format prompt for better Qwen output.

    Returns:
    - str: The COT prompt.
    """
    if dataset_name in ["wtq", "wikitablequestion"]:
        if use_strict_format:
            from prompt.wtq.cot_strict import cot_prompt_strict
            return cot_prompt_strict
        else:
            from prompt.wtq.cot import cot_prompt
            return cot_prompt
    elif dataset_name in ["tabfact", "tabularfact"]:
        from prompt.tabfact.cot import cot_prompt
        return cot_prompt
    elif dataset_name in ["train", "valid"]:
        if use_strict_format:
            from prompt.wtq.cot_strict import cot_prompt_strict
            return cot_prompt_strict
        else:
            from prompt.wtq.cot import cot_prompt
            return cot_prompt
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")





def query(model, long_model, prompt, temperature, self_consistency, system=None, enable_thinking: bool = False):
    """
    Execute a query on the model and handle prompt length for choosing the appropriate model.

    Args:
    - model: The primary model for querying.
    - long_model: The long version of the model for longer prompts.
    - prompt (str): The prompt to query.
    - temperature (float): The temperature setting for the query.
    - self_consistency (int): The number of outputs to generate.

    Returns:
    - Tuple: (text, response)
    """

    prompt_length = len(long_model.tokenizer.encode(prompt))

    # 提高生成配额以减少截断：
    # - "4K" 分支：给更高的总窗口预算（默认 8192），保证最小生成 512
    # - "16K" 分支：给更高的总窗口预算（默认 28672），保证最小生成 1024
    # 注意：这里的预算仅用于计算 max_new_tokens，上下文真实上限由具体模型决定
    SHORT_TOTAL_BUDGET = 8192   # 原先逻辑约 4000
    LONG_TOTAL_BUDGET = 28672   # 原先逻辑约 15360
    MIN_SHORT_GEN = 512
    MIN_LONG_GEN = 1024

    if isinstance(model, Model):
        if prompt_length <= 3328:  # 4K上下文（提升生成预算）
            try:
                gen_budget = max(
                    MIN_SHORT_GEN, SHORT_TOTAL_BUDGET - prompt_length)
                return model.query(prompt=prompt, temperature=temperature, max_tokens=gen_budget, n=self_consistency, system=system, enable_thinking=enable_thinking)
            except Exception as e:
                print(f"Error in 4K model query: {e}")
                # Fallback to 16K model
                print(
                    f"Prompt length -- {prompt_length} is too long or 4K model failed, we use the long-context version.")
                try:
                    gen_budget = max(
                        MIN_LONG_GEN, LONG_TOTAL_BUDGET - prompt_length)
                    return long_model.query(prompt=prompt, temperature=temperature, max_tokens=gen_budget, n=self_consistency, system=system, enable_thinking=enable_thinking)
                except Exception as e2:
                    print(f"Error in 16K model query: {e2}")
                    # Return None to indicate failure
                    if self_consistency == 1:
                        return None, None
                    else:
                        return [None] * self_consistency, None
        elif prompt_length <= 14592:  # 16K上下文（提升生成预算）
            print(
                f"Prompt length -- {prompt_length} is long, we use the long-context version.")
            try:
                gen_budget = max(
                    MIN_LONG_GEN, LONG_TOTAL_BUDGET - prompt_length)
                return long_model.query(prompt=prompt, temperature=temperature, max_tokens=gen_budget, n=self_consistency, system=system, enable_thinking=enable_thinking)
            except Exception as e:
                print(f"Error in 16K model query: {e}")
                # Return None to indicate failure
                if self_consistency == 1:
                    return None, None
                else:
                    return [None] * self_consistency, None
        else:
            if self_consistency == 1:
                return f"Prompt length -- {prompt_length} is too long", {prompt_length: prompt_length}
            else:
                return ["Prompt length -- {prompt_length} is too long"] * self_consistency, {prompt_length: prompt_length}
    else:
        # no short version of the model provided, which means we use the long version for all prompts
        if prompt_length <= 14592:
            try:
                return long_model.query(prompt=prompt, temperature=temperature, max_tokens=15360 - prompt_length, n=self_consistency, system=system, enable_thinking=enable_thinking)
            except Exception as e:
                print(f"Error in long model query: {e}")
                # Return None to indicate failure
                if self_consistency == 1:
                    return None, None
                else:
                    return [None] * self_consistency, None
        else:
            if self_consistency == 1:
                return f"Prompt length -- {prompt_length} is too long", {prompt_length: prompt_length}
            else:
                return ["Prompt length -- {prompt_length} is too long"] * self_consistency, {prompt_length: prompt_length}


def check_transpose(model: Model, long_model: Model, table, title, table_id, perturbation, transpose_cache, norm_cache, cache_dir):
    """
    Check if the table needs transposing, using cache if available.

    Args:
    - model, long_model (Model): The models used for querying.
    - table (str): The markdown representation of the table. 表格数据
    - title (str): The title of the table. 标题
    - table_id (str): The ID of the table. 表格ID
    - perturbation (str): The perturbation applied to the table. 扰动类型
    - transpose_cache (dict): Cache for transpose information. 转置缓存
    - norm_cache (bool): Flag to determine if normalization caching is enabled. 是否启用缓存
    - cache_dir (str): Directory for caching. 缓存目录

    Returns:
    - bool: Whether the table needs transposing.
    """
    from prompt.general.transpose_check import header_check_prompt

    # Check cache first
    if table_id in transpose_cache and perturbation in transpose_cache[table_id]:
        return transpose_cache[table_id][perturbation]

    try:
        # 添加表格数据的有效性检查
        if not table or not isinstance(table, str):
            print(f"Warning: Invalid table data for table {table_id}")
            return False

        table_rows = table.split("\n")
        if not table_rows or len(table_rows) == 0:
            print(f"Warning: Empty table data for table {table_id}")
            return False

        # 添加边界检查，确保第一行存在且有效
        if len(table_rows) > 0:
            first_row_cells = table_rows[0].split("|")
            if len(first_row_cells) > 2:  # 至少需要有一个有效单元格
                first_row = ", ".join([cell.strip()
                                      for cell in first_row_cells[1:-1]])
            else:
                first_row = ""
        else:
            first_row = ""

        # 添加边界检查，确保第一列存在且有效
        first_column_cells = []
        for row in table_rows:
            row_cells = row.split("|")
            if len(row_cells) > 1:  # 确保至少有一个分隔符
                first_column_cells.append(row_cells[1].strip())
        first_column = ", ".join(first_column_cells).strip()

        # Construct and send the query
        transpose_check_prompt = header_check_prompt.replace("[TABLE]", table)\
            .replace("[FIRST_ROW]", first_row)\
            .replace("[FIRST_COLUMN]", first_column)\
            .replace("[TITLE]", title)\
            .strip()

        text, _ = query(model, long_model, transpose_check_prompt,
                        temperature=0, self_consistency=1)

        # 添加text的有效性检查，确保text不为None且是有效字符串
        if text is None or not isinstance(text, str):
            print(
                f"Warning: Model query returned invalid result for table {table_id}: {text}")
            transpose_flag = False  # 默认不转置
        else:
            transpose_flag = parse_header_checking_result(text)

    except Exception as e:
        print(
            f"Error occurred while checking transpose for table {table_id}: {e}")
        transpose_flag = False  # 默认不转置

    # Update cache if necessary
    if norm_cache:
        if table_id not in transpose_cache:
            transpose_cache[table_id] = {}
        transpose_cache[table_id][perturbation] = transpose_flag
        with open(os.path.join(cache_dir, "transpose.json"), "w", encoding='utf-8') as f:
            json.dump(transpose_cache, f, indent=4)

    return transpose_flag


def check_sort(model: Model, long_model: Model, df, title, table_id, perturbation, resort_cache, norm_cache, cache_dir):
    """
    Check if the table needs sorting, using cache if available.

    Args:
    - model, long_model: The models used for querying.
    - df (DataFrame): The DataFrame representation of the table.
    - title (str): The title of the table.
    - table_id (str): The ID of the table.
    - perturbation (str): The perturbation applied to the table.
    - resort_cache (dict): Cache for sorting information.
    - norm_cache (bool): Flag to determine if normalization caching is enabled.
    - cache_dir (str): Directory for caching.

    Returns:
    - List: The list of columns for sorting.
    """
    from prompt.general.resort_check import sort_prompt

    # Check cache first
    if table_id in resort_cache and perturbation in resort_cache[table_id]:
        return resort_cache[table_id][perturbation]

    try:
        # 添加DataFrame的有效性检查
        if df is None or df.empty:
            print(f"Warning: Invalid or empty DataFrame for table {table_id}")
            return []

        # Construct and send the query
        partial_table = print_partial_markdown(df)

        # 添加边界检查，确保表格数据有效
        if not partial_table or not isinstance(partial_table, str):
            print(f"Warning: Invalid partial table data for table {table_id}")
            return []

        table_rows = partial_table.split("\n")
        if not table_rows or len(table_rows) == 0:
            print(f"Warning: Empty partial table data for table {table_id}")
            return []

        # 添加边界检查，确保第一行存在且有效
        if len(table_rows) > 0:
            first_row_cells = table_rows[0].split("|")
            if len(first_row_cells) > 2:  # 至少需要有一个有效单元格
                heading_list = [cell.strip() for cell in first_row_cells[1:-1]]
            else:
                heading_list = []
        else:
            heading_list = []

        headings = "; ".join(heading_list)

        resort_check_prompt = sort_prompt.replace("[TABLE]", partial_table)\
            .replace("[HEADINGS]", headings)\
            .replace("[TITLE]", title)\
            .strip()

        text, _ = query(model, long_model, resort_check_prompt,
                        temperature=0, self_consistency=1)

        # 添加text的有效性检查，确保text不为None且是有效字符串
        if text is None or not isinstance(text, str):
            print(
                f"Warning: Model query returned invalid result for table {table_id}: {text}")
            resort_list = []  # 默认不排序
        else:
            resort_list = parse_header_sorting_result(text)
            if resort_list is None:  # 处理parse_header_sorting_result返回None的情况
                resort_list = []

    except Exception as e:
        print(f"Error occurred while checking sort for table {table_id}: {e}")
        resort_list = []  # 默认不排序

    # Update cache if necessary
    if norm_cache:
        os.makedirs(cache_dir, exist_ok=True)
        if table_id not in resort_cache:
            resort_cache[table_id] = {}
        resort_cache[table_id][perturbation] = resort_list
        with open(os.path.join(cache_dir, "resort.json"), "w", encoding='utf-8') as f:
            json.dump(resort_cache, f, indent=4)

    return resort_list


def read_json_file(file_path):
    """
    Read a JSON file.

    Args:
    - file_path (str): The path to the JSON file.

    Returns:
    - dict: The JSON file.
    """
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except:
        return {}

    return data
