"""Gold-blind consensus fusion for three text solvers and an optional Python solver."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluation.metrics import summarize_records
from paper1_runtime.adapters import get_adapter


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def invalid(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = re.sub(r"(?i)^final\s+answer\s*:\s*", "", value).strip().lower()
        return not text or text in {"none", "null", "nan", "n/a", "error", "unknown"}
    return False


def norm(adapter, value: Any) -> str:
    value = adapter.normalize_prediction(value)
    if isinstance(value, (list, tuple)):
        return " | ".join(str(item).strip().lower() for item in value)
    return "" if value is None else str(value).strip().lower()


def check_alignment(sources: list[list[dict[str, Any]]]) -> None:
    if len({len(rows) for rows in sources}) != 1:
        raise ValueError("source row counts differ")
    ids = [[row["question_id"] for row in rows] for rows in sources]
    if any(current != ids[0] for current in ids[1:]):
        raise ValueError("sources are not aligned by question_id")


def run(args: argparse.Namespace) -> int:
    text_sources = [
        read_rows(Path(path))
        for path in (args.dp_baseline, args.dp_optimized, args.dp_structured)
    ]
    python_rows = read_rows(Path(args.python_source)) if args.python_source else None
    check_alignment(text_sources + ([python_rows] if python_rows is not None else []))
    adapter = get_adapter(args.dataset, sub_sample=False)
    output = HERE / "outputs" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.jsonl"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(result_path)
    if result_path.exists():
        result_path.unlink()
    config = {
        **vars(args),
        "gold_excluded_from_selection": True,
        "policy": (
            "three-text normalized majority; baseline tie-break; Python changes an all-different "
            "text tie only when it agrees with an alternative text solver"
        ),
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    for index, (base, optimized, structured) in enumerate(zip(*text_sources)):
        candidates = [
            base.get("final_answer"),
            optimized.get(args.dp_optimized_field),
            structured.get("final_answer"),
        ]
        normals = [norm(adapter, value) for value in candidates]
        valid_indices = [
            position
            for position, value in enumerate(candidates)
            if normals[position] and not invalid(value)
        ]
        counts = Counter(normals[position] for position in valid_indices)
        majority_norm, majority_count = counts.most_common(1)[0] if counts else ("", 0)
        labels = ["dp_baseline", "dp_optimized", "dp_structured"]
        if majority_count >= 2:
            chosen_index = next(
                position for position in valid_indices if normals[position] == majority_norm
            )
            final_answer = candidates[chosen_index]
            mode = "text_majority"
            selected = labels[chosen_index]
        elif 0 in valid_indices:
            chosen_index = 0
            final_answer = candidates[0]
            mode = "baseline_tie_break"
            selected = "dp_baseline"
        elif valid_indices:
            chosen_index = valid_indices[0]
            final_answer = candidates[chosen_index]
            mode = "valid_text_fallback"
            selected = labels[chosen_index]
        else:
            chosen_index = -1
            final_answer = None
            mode = "no_valid_text"
            selected = "none"

        python_answer = python_rows[index].get("final_answer") if python_rows is not None else None
        python_norm = norm(adapter, python_answer)
        if python_norm and python_norm == norm(adapter, final_answer):
            mode += "+python_agreement"
        elif majority_count < 2 and python_norm and not invalid(python_answer):
            alternative_matches = [
                position for position in (1, 2) if normals[position] == python_norm
            ]
            if alternative_matches:
                chosen_index = alternative_matches[0]
                final_answer = candidates[chosen_index]
                selected = labels[chosen_index]
                mode = "cross_modal_tie_break"

        gold = base["answer"]
        correct = adapter.is_correct(final_answer, gold)
        record = {
            "idx": base["idx"],
            "answer": gold,
            "text": "",
            "transpose": base.get("transpose", False),
            "resort": base.get("resort", []),
            "question_id": base["question_id"],
            "table_id": base.get("table_id"),
            "title": base.get("title", ""),
            "table": base["table"],
            "question": base["question"],
            "sample_id": base.get("sample_id", base["question_id"]),
            "method": "consensus_joint",
            "candidate_dp_baseline": candidates[0],
            "candidate_dp_optimized": candidates[1],
            "candidate_dp_structured": candidates[2],
            "candidate_python": python_answer,
            "normalized_candidates": {
                "dp_baseline": normals[0],
                "dp_optimized": normals[1],
                "dp_structured": normals[2],
                "python": python_norm,
            },
            "text_majority_count": majority_count,
            "selected_method": selected,
            "selection_mode": mode,
            "final_answer": final_answer,
            "normalized_prediction": adapter.normalize_prediction(final_answer),
            "correct": bool(correct),
            "api_calls": 0,
            "gold_excluded_from_selection": True,
        }
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        records.append(record)
        print(
            base["idx"],
            base["question_id"],
            selected,
            mode,
            "correct=",
            bool(correct),
            flush=True,
        )

    (output / "summary.json").write_text(
        json.dumps(summarize_records(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wtq", "tabfact"], required=True)
    parser.add_argument("--dp-baseline", required=True)
    parser.add_argument("--dp-optimized", required=True)
    parser.add_argument("--dp-optimized-field", default="final_answer")
    parser.add_argument("--dp-structured", required=True)
    parser.add_argument("--python-source")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--overwrite", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
