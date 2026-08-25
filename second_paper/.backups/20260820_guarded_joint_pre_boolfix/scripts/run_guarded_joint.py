#!/usr/bin/env python3
"""Gold-blind guarded fusion for DP and audited Python predictions.

The script treats the direct-prompting prediction as the protected primary
answer.  A blind double-verifier proposal may replace it only when generic,
development-calibrated safety checks pass.  WTQ additionally has a deterministic
duration rule that is activated only when two independently produced component
answers agree on the numeric span.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def answer_text(row: dict[str, Any]) -> str:
    value = row.get("final_answer", row.get("normalized_prediction", ""))
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "").strip()


def numeric_signature(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?<!\w)[+-]?(?:\d+(?:,\d{3})*|\d*\.\d+)", value.replace(",", "")))


def is_pure_numeric(value: str) -> bool:
    return bool(re.fullmatch(r"\s*[+-]?(?:\d+(?:,\d{3})*|\d*\.\d+)\s*%?\s*", value))


def aligned_rows(paths: list[Path]) -> list[list[dict[str, Any]]]:
    streams = [load_jsonl(path) for path in paths]
    lengths = {len(stream) for stream in streams}
    if len(lengths) != 1:
        raise ValueError(f"row count mismatch: {[len(stream) for stream in streams]}")
    for position, group in enumerate(zip(*streams)):
        ids = {str(row.get("question_id", row.get("sample_id", row.get("idx")))) for row in group}
        indices = {row.get("idx") for row in group}
        if len(ids) != 1 or len(indices) != 1:
            raise ValueError(f"source alignment mismatch at position {position}: ids={ids}, idx={indices}")
    return streams


def python_is_usable(row: dict[str, Any]) -> tuple[bool, str]:
    if not row.get("final_execution_success", row.get("execution_success", False)):
        return False, "python_execution_failed"
    if row.get("execution_warnings"):
        return False, "python_execution_warning"
    status = str(row.get("audit_status", ""))
    if status == "passed":
        return True, "python_audit_passed"

    question = str(row.get("question", "")).lower()
    operations = (
        row.get("question_intent_ir", {}).get("operation_sequence", [])
        if isinstance(row.get("question_intent_ir"), dict)
        else []
    )
    sequence_exception = (
        "consecutive" in question
        and any(str(op).lower() == "sequence_check" for op in operations)
    )
    if sequence_exception:
        return True, "audited_sequence_exception"
    return False, f"python_audit_{status or 'unknown'}"


def decide_wtq(
    primary: dict[str, Any],
    blind: dict[str, Any],
    python_row: dict[str, Any],
    optimized_dp: dict[str, Any],
) -> tuple[str, str, bool, list[str]]:
    primary_answer = answer_text(primary)
    blind_answer = answer_text(blind)
    python_answer = answer_text(python_row)
    optimized_dp_answer = answer_text(optimized_dp)
    question = str(primary.get("question", "")).strip().lower()
    trace: list[str] = []

    # A duration is an elapsed difference, not an inclusive row count.  Require
    # independent Python and optimized-DP numeric agreement before applying it.
    duration_question = bool(
        re.search(r"\bhow long\b|\bhow many\s+(?:years?|months?|days?|hours?|minutes?|seconds?)\b", question)
    )
    duration_unit = bool(re.search(r"\b(?:years?|months?|days?|hours?|minutes?|seconds?)\b", optimized_dp_answer.lower()))
    py_nums = numeric_signature(python_answer)
    dp_nums = numeric_signature(optimized_dp_answer)
    if (
        duration_question
        and duration_unit
        and python_row.get("final_execution_success", python_row.get("execution_success", False))
        and py_nums
        and py_nums == dp_nums
    ):
        trace.append("duration_cross_modal_numeric_agreement")
        return optimized_dp_answer, "optimized_dp_duration", bool(optimized_dp.get("correct", False)), trace

    if not blind.get("override_accepted", False):
        trace.append("blind_verifier_kept_primary")
        return primary_answer, "dp_baseline", bool(primary.get("correct", False)), trace

    if (
        re.match(r"^(?:which|who)\b", question)
        and is_pure_numeric(blind_answer)
        and not is_pure_numeric(primary_answer)
    ):
        trace.append("blocked_entity_to_numeric_type_change")
        return primary_answer, "dp_baseline", bool(primary.get("correct", False)), trace

    trace.append("blind_double_verifier_override")
    return blind_answer, "blind_double_verifier", bool(blind.get("correct", False)), trace


def decide_tabfact(
    primary: dict[str, Any],
    blind: dict[str, Any],
    python_row: dict[str, Any],
    optimized_dp: dict[str, Any],
) -> tuple[str, str, bool, list[str]]:
    del optimized_dp
    primary_answer = answer_text(primary)
    blind_answer = answer_text(blind)
    question = str(primary.get("question", "")).strip().lower()
    trace: list[str] = []

    if not blind.get("override_accepted", False):
        trace.append("blind_verifier_kept_primary")
        return primary_answer, "dp_baseline", bool(primary.get("correct", False)), trace

    usable, reason = python_is_usable(python_row)
    if not usable:
        trace.append(f"blocked_{reason}")
        return primary_answer, "dp_baseline", bool(primary.get("correct", False)), trace

    high_risk_patterns = [
        (r"\b\d+\s*/\s*\d+\b", "explicit_fraction"),
        (r"\bout of\b", "named_ratio"),
        (r"\byears?\s+(?:before|after)\b", "season_interval_arithmetic"),
    ]
    for pattern, name in high_risk_patterns:
        if re.search(pattern, question):
            trace.append(f"blocked_{name}")
            return primary_answer, "dp_baseline", bool(primary.get("correct", False)), trace

    trace.extend([reason, "blind_double_verifier_override"])
    return blind_answer, "blind_double_verifier", bool(blind.get("correct", False)), trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("wtq", "tabfact"), required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--python", dest="python_path", type=Path, required=True)
    parser.add_argument("--optimized-dp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    primary_rows, blind_rows, python_rows, optimized_dp_rows = aligned_rows(
        [args.primary, args.blind, args.python_path, args.optimized_dp]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    written: list[dict[str, Any]] = []
    chooser = decide_wtq if args.dataset == "wtq" else decide_tabfact
    for primary, blind, python_row, optimized_dp in zip(
        primary_rows, blind_rows, python_rows, optimized_dp_rows
    ):
        chosen, selected_source, correct, trace = chooser(primary, blind, python_row, optimized_dp)
        counts[selected_source] += 1
        counts["correct" if correct else "incorrect"] += 1
        record = dict(primary)
        record.update(
            {
                "method": "guarded_joint",
                "selected_method": selected_source,
                "selection_mode": "guarded_override" if selected_source != "dp_baseline" else "protected_primary",
                "final_answer": chosen,
                "normalized_prediction": chosen,
                "correct": correct,
                "guard_trace": trace,
                "gold_excluded_from_selection": True,
                "source_answers": {
                    "dp_baseline": answer_text(primary),
                    "blind_double_verifier": answer_text(blind),
                    "optimized_python": answer_text(python_row),
                    "optimized_dp": answer_text(optimized_dp),
                },
            }
        )
        # Required stable order: idx, gold answer, then all diagnostic fields.
        ordered = {"idx": record.pop("idx"), "answer": record.pop("answer", [])}
        ordered.update(record)
        written.append(ordered)

    with args.output.open("w", encoding="utf-8") as handle:
        for row in written:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset": args.dataset,
        "rows": len(written),
        "accuracy": counts["correct"] / len(written) if written else 0.0,
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "selected_source_counts": {
            key: value for key, value in counts.items() if key not in {"correct", "incorrect"}
        },
        "gold_excluded_from_selection": True,
    }
    summary_path = args.summary or args.output.with_name("summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
