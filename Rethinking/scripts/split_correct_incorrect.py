import json
import argparse
from typing import Tuple
import sys
import os

# Ensure project root is on sys.path for `utils` imports
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from utils.eval import extract_answer, eval_ex_match
except Exception:
    # Fallback: load module directly from file in project utils folder
    import importlib.util

    util_path = os.path.join(PROJECT_ROOT, "utils", "eval.py")
    spec = importlib.util.spec_from_file_location("utils.eval", util_path)
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)
    extract_answer = getattr(eval_mod, "extract_answer")
    eval_ex_match = getattr(eval_mod, "eval_ex_match")


def judge(example: dict) -> Tuple[bool, str, str]:
    """
    Return (is_correct, gold, pred) for a single example.
    Assumes fields: "answer" (list[str]) and "text" (str or list[str]).
    """
    # gold can be a list of strings, join with comma-space like evaluate.py
    gold_list = example.get("answer", [])
    gold = ", ".join(gold_list) if isinstance(
        gold_list, list) else str(gold_list)

    text = example.get("text", "")
    if isinstance(text, list):
        # If it's a list, take the last one (usually the final turn) or the first non-empty
        for t in reversed(text):
            if isinstance(t, str) and t.strip():
                text = t
                break
        if isinstance(text, list):
            text = ""

    pred = extract_answer(text) if isinstance(text, str) else None
    if pred is None:
        return False, gold, ""

    ok = bool(eval_ex_match(gold, pred))
    return ok, gold, pred


def main():
    parser = argparse.ArgumentParser(
        description="Split JSONL results by correctness using project eval logic.")
    parser.add_argument("input", help="Path to input result.jsonl")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (defaults to input's folder)")
    parser.add_argument("--correct_name", default="correct.jsonl",
                        help="Filename for correct outputs")
    parser.add_argument("--incorrect_name", default="incorrect.jsonl",
                        help="Filename for incorrect outputs")
    args = parser.parse_args()

    in_path = args.input

    # If user accidentally concatenated the filename twice like
    # `result.jsonlresult.jsonl`, try to detect a nearby existing filename
    # by progressively trimming suffixes up to the first occurrence of
    # the original basename. If the exact path exists, keep it.
    if not os.path.exists(in_path):
        # try detect duplicated basename pattern
        base = os.path.basename(in_path)
        dirname = os.path.dirname(in_path) or "."
        # attempt to find any file in dirname that is a suffix of the given path
        candidates = []
        try:
            for fn in os.listdir(dirname):
                fn_path = os.path.join(dirname, fn)
                if not os.path.isfile(fn_path):
                    continue
                if in_path.endswith(fn):
                    candidates.append(fn_path)
        except FileNotFoundError:
            candidates = []

        if len(candidates) == 1:
            fixed = candidates[0]
            print(
                f"Input path '{in_path}' not found. Using detected file '{fixed}'.")
            in_path = fixed
        elif len(candidates) > 1:
            print(
                f"Input path '{in_path}' not found. Multiple candidate files found in '{dirname}':")
            for c in candidates:
                print("  ", c)
            print("Please re-run with the correct input path.")
            return
        else:
            print(
                f"Input path '{in_path}' not found and no candidate file detected in '{dirname}'.")
            return

    if args.outdir is None:
        outdir = os.path.dirname(in_path) or "."
    else:
        outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)
    correct_path = os.path.join(outdir, args.correct_name)
    incorrect_path = os.path.join(outdir, args.incorrect_name)

    total = 0
    n_correct = 0
    n_incorrect = 0

    # Read all examples, judge them, collect into lists, then sort by idx (if present)
    examples_ok = []
    examples_bad = []

    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                # Skip malformed lines
                continue

            total += 1
            ok, gold, pred = judge(ex)
            # store diagnostics
            ex["pred_extracted"] = pred
            ex["gold_joined"] = ", ".join(ex.get("answer", [])) if isinstance(
                ex.get("answer", []), list) else str(ex.get("answer"))
            ex["is_correct"] = ok

            # preserve original index order if idx missing; otherwise rely on numeric idx
            if ok:
                examples_ok.append(ex)
            else:
                examples_bad.append(ex)

    def _get_idx(item):
        # try common idx keys
        for k in ("idx", "id", "index"):
            if k in item:
                try:
                    return int(item[k])
                except Exception:
                    try:
                        return int(str(item[k]))
                    except Exception:
                        return None
        return None

    # If any items have idx, sort by it; otherwise keep original order
    if any(_get_idx(x) is not None for x in examples_ok):
        examples_ok.sort(key=lambda x: (
            _get_idx(x) if _get_idx(x) is not None else float('inf')))
    if any(_get_idx(x) is not None for x in examples_bad):
        examples_bad.sort(key=lambda x: (
            _get_idx(x) if _get_idx(x) is not None else float('inf')))

    # Write out
    with open(correct_path, "w", encoding="utf-8") as f_ok, open(incorrect_path, "w", encoding="utf-8") as f_bad:
        for ex in examples_ok:
            n_correct += 1
            f_ok.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for ex in examples_bad:
            n_incorrect += 1
            f_bad.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(
        f"Done. Total: {total}, Correct: {n_correct}, Incorrect: {n_incorrect}")
    print(f"Correct file: {correct_path}")
    print(f"Incorrect file: {incorrect_path}")


if __name__ == "__main__":
    main()
