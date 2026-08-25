from collections import Counter
from utils.eval import eval_ex_match, extract_answer
import random
import json
import numpy as np
from tqdm import tqdm
from fire import Fire
from typing import Union, List, Tuple
import os


def flatten(lst):
    flat_list = []
    for i in lst:
        if isinstance(i, list):
            flat_list.extend(flatten(i))
        else:
            flat_list.append(i)
    return flat_list


def load_results(checkpoints, elements_per_checkpoint):

    print(f"Loading {checkpoints}...")
    # not a list or a tuple, make it a list
    if not isinstance(checkpoints, list) and not isinstance(checkpoints, tuple):
        # try to split by comma
        if "," in checkpoints:
            checkpoints = checkpoints.split(",")
            # remove the spaces
            checkpoints = [checkpoint.strip() for checkpoint in checkpoints]
        else:
            checkpoints = [checkpoints]

    all_results = []

    # read all checkpoints
    for checkpoint in checkpoints:
        print(f"Loading {checkpoint}...")

        if checkpoint.endswith(".jsonl"):
            with open(checkpoint, "r", encoding="utf-8") as f:
                results = [json.loads(line) for line in f.readlines()]
        else:
            with open(f"output/{checkpoint}/result.jsonl", "r", encoding="utf-8") as f:
                results = [json.loads(line) for line in f.readlines()]

        print(f"Loaded {len(results)} results.")

        # deduplicate the results by id
        results = {result["question_id"]: result for result in results}
        results = list(results.values())

        all_results.append(results)

    # make sure the checkpoints are same length, if not, cut the longer one
    min_len = min([len(results) for results in all_results])
    all_results = [results[:min_len] for results in all_results]

    # the results are now in the form of [[dict, dict, ...], [dict, dict, ...], ...]
    # we want to combine them into one list of dicts by aggregating the dict["text"] field
    combined_results = []
    for i, results in enumerate(all_results):
        if i == 0:
            # if this is the first checkpoint, just add the results
            combined_results = results
            # make the text field a list of list
            for result in combined_results:
                # random sample the text field if specified
                if isinstance(result["text"], str):
                    result["text"] = [result["text"]]
                try:
                    result["text"] = random.sample(
                        result["text"], elements_per_checkpoint[i]) if elements_per_checkpoint else [result["text"]]
                except Exception as e:
                    result["text"] = random.sample(
                        result["text"], elements_per_checkpoint[i]) if elements_per_checkpoint else [result["text"]]

        else:
            # if this is not the first checkpoint, add the text field to the existing list
            for j, result in enumerate(results):
                # remember to random sample the text field if specified
                if isinstance(result["text"], str):
                    result["text"] = [result["text"]]
                temp = random.sample(
                    result["text"], elements_per_checkpoint[i]) if elements_per_checkpoint else result["text"]

                # add by question id instead of index
                for k, combined_result in enumerate(combined_results):
                    if combined_result["question_id"] == result["question_id"]:
                        combined_results[k]["text"].append(temp)
                        break

    # now we have a list of dicts with the text field being a list of list
    return combined_results


def normalize_tabfact_answer(answer):
    """Convert TabFact answer to standardized format.
    Input can be: 0/1 (int), "0"/"1" (str), "yes"/"no" (str), "true"/"false" (str)
    Output: "0" or "1" (str)
    """
    answer_str = str(answer).lower().strip()
    if answer_str in ['1', 'yes', 'true']:
        return '1'
    elif answer_str in ['0', 'no', 'false']:
        return '0'
    return answer_str


def eval_wtq(checkpoints: Union[List, Tuple, str], elements_per_checkpoint: Union[None, int, List] = None,
             n_times: int = 100, sub_sample_question_ids: list = None, output_error_indices: bool = False):
    print("Starting Evaluation...")

    results = load_results(checkpoints, elements_per_checkpoint)
    acc_list = []
    all_error_indices = []  # Store error indices for each trial

    for i in tqdm(range(n_times), desc="Progress", unit="batch"):
        acc, total = 0, 0
        error_indices = []  # Track errors in this trial

        for result in results:
            if sub_sample_question_ids and result["question_id"] not in sub_sample_question_ids:
                continue

            # Handle both list and single value answers
            if isinstance(result["answer"], list):
                answer = ", ".join(str(a) for a in result["answer"])
            else:
                answer = normalize_tabfact_answer(result["answer"])

            if isinstance(result["text"], str):
                result["text"] = [result["text"]]

            # Flatten the list to make sure it is 1D
            result["text"] = flatten(result["text"])

            preds = [extract_answer(text) for text in result["text"]]
            # Normalize predictions to 0/1 format
            preds = [normalize_tabfact_answer(pred) for pred in preds if pred]

            if n_times > 1:
                np.random.shuffle(preds)
            if not preds:
                error_indices.append(result.get(
                    "idx", result.get("question_id")))
                total += 1
                continue

            # Majority voting
            pred_count = Counter(preds)
            pred, _ = pred_count.most_common(1)[0]

            if eval_ex_match(answer, pred):
                acc += 1
            else:
                error_indices.append(result.get(
                    "idx", result.get("question_id")))
            total += 1

        acc_list.append(acc / total * 100)
        all_error_indices.append(error_indices)

    print("Evaluation Complete.")

    # Print results
    print(
        f"Statistical Summary of {n_times} Trials on {total} Examples from Combined Checkpoints")
    print(
        f"Min Accuracy: {min(acc_list):.2f}% ({np.round(min(acc_list) / 100 * total)}/{total})")
    print(
        f"Max Accuracy: {max(acc_list):.2f}% ({np.round(max(acc_list) / 100 * total)}/{total})")
    print(
        f"Mean Accuracy: {np.mean(acc_list):.2f}% ({np.round(np.mean(acc_list) / 100 * total)}/{total})")
    print(f"Standard Deviation: {np.std(acc_list):.2f}%")

    # Output error indices if requested
    if output_error_indices and n_times == 1:
        print(
            f"\nError Indices (Total: {len(all_error_indices[0])}/{total}):")
        print(all_error_indices[0])
    elif output_error_indices and n_times > 1:
        # Find common errors across all trials
        common_errors = set(all_error_indices[0])
        for errors in all_error_indices[1:]:
            common_errors &= set(errors)
        print(
            f"\nCommon Error Indices across all {n_times} trials (Total: {len(common_errors)}/{total}):")
        print(sorted(list(common_errors)))

# Example usage:
# eval_wtq(checkpoints="checkpoint1.jsonl")
# eval_wtq(checkpoints=["checkpoint1.jsonl", "checkpoint2.jsonl"], elements_per_checkpoint=[5, 5])


def eval_tabfact(checkpoints: Union[List, Tuple, str], elements_per_checkpoint: Union[None, int, List] = None,
                 n_times: int = 100, sub_sample_question_ids: list = None, output_error_indices: bool = False):
    """Deprecated: Use eval_wtq instead. This function is kept for backward compatibility."""
    return eval_wtq(checkpoints, elements_per_checkpoint, n_times, sub_sample_question_ids, output_error_indices)


if __name__ == "__main__":
    Fire(eval_wtq)
