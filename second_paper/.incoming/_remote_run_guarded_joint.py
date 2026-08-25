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
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    # False is a valid TabFact prediction and must not collapse to an empty string.
    return str(value).strip()


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


def tabfact_boolean(value: str) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    return None


def tabfact_semantic_route(question: str, table: str) -> str | None:
    """Identify the same narrow semantic routes used by the optimized prompts."""
    q = question.lower()
    header = table.splitlines()[0].lower() if table else ""
    category_comparison = bool(
        re.search(r"\b(?:object type|category|categories)\b", q)
        and re.search(r"\b(?:more|less|greater|fewer)\b", q)
        and not re.search(r"\b(?:all|every|average|mean)\b", q)
    )
    if category_comparison:
        return "category_pair_comparison"

    asks_points = bool(re.search(r"\bpoints?\b", q))
    asks_goals = bool(re.search(r"\bgoals?\b", q))
    score_header = bool(re.search(r"\b(?:score|agg)\b", header))
    goals_header = bool(re.search(r"\bgoals?\b", header))
    points_header = bool(re.search(r"\bpoints?\b", header))
    if (asks_points and (score_header or goals_header) and not points_header) or (
        asks_goals and score_header and not goals_header
    ):
        return "sports_score_alias"
    return None


def wtq_competition_count_route(question: str) -> bool:
    q = question.lower()
    count_cue = bool(
        re.search(r"\b(?:how many|number of|total number|how many total)\b", q)
    )
    competition = bool(re.search(r"\bcompetitions?\b", q))
    participation = bool(
        competition
        and re.search(
            r"\b(?:compete(?:d|s)?|competing|participate(?:d|s)?|participating|play(?:ed|s|ing)?)\s+in\b",
            q,
        )
    )
    return count_cue and competition and not participation


def wtq_film_award_route(question: str, table: str) -> bool:
    q = question.lower()
    header = table.splitlines()[0].lower() if table else ""
    return bool(
        re.search(r"\b(?:how many|number of|total number|how many total)\b", q)
        and re.search(r"\bfilms?\b", q)
        and re.search(r"\b(?:nominated|nomination)\b", q)
        and re.search(r"\bawards?\b", q)
        and re.search(r"\bnotes?\b", header)
    )


def verified_negative_row_count(row: dict[str, Any], answer: str) -> bool:
    code_ir = row.get("code_intent_ir")
    if not isinstance(code_ir, dict):
        return False
    if code_ir.get("aggregation") != "count" or code_ir.get("return_type") != "number":
        return False
    if not any(spec.get("op") == "!=" for spec in code_ir.get("filters", [])):
        return False
    answer_numbers = numeric_signature(answer)
    observed_numbers = numeric_signature(str(code_ir.get("observed_value", "")))
    if not answer_numbers or answer_numbers != observed_numbers:
        return False
    return any(
        event.get("operation") == "filter"
        and numeric_signature(str(event.get("rows_after", ""))) == answer_numbers
        for event in code_ir.get("runtime_evidence", [])
        if isinstance(event, dict)
    )


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

    usable, usable_reason = python_is_usable(python_row)
    routed_count = wtq_competition_count_route(question)
    py_dp_agreement = bool(py_nums and py_nums == dp_nums)
    negative_row_evidence = bool(
        re.search(r"\b(?:not\s+in|outside|excluding)\b", question)
        and verified_negative_row_count(python_row, python_answer)
    )
    if (
        routed_count
        and usable
        and is_pure_numeric(python_answer)
        and numeric_signature(primary_answer) != py_nums
        and (py_dp_agreement or negative_row_evidence)
    ):
        trace.extend(
            [
                usable_reason,
                "routed_competition_row_count",
                "python_dp_agreement" if py_dp_agreement else "verified_negative_row_count",
            ]
        )
        return (
            python_answer,
            "optimized_semantic_consensus",
            bool(python_row.get("correct", False)),
            trace,
        )

    film_award_route = wtq_film_award_route(
        question, str(primary.get("table", ""))
    )
    if (
        film_award_route
        and usable
        and is_pure_numeric(python_answer)
        and py_dp_agreement
        and numeric_signature(primary_answer) != py_nums
    ):
        trace.extend(
            [usable_reason, "routed_film_award_count", "python_dp_agreement"]
        )
        return (
            python_answer,
            "optimized_semantic_consensus",
            bool(python_row.get("correct", False)),
            trace,
        )

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
    primary_answer = answer_text(primary)
    blind_answer = answer_text(blind)
    python_answer = answer_text(python_row)
    optimized_dp_answer = answer_text(optimized_dp)
    question = str(primary.get("question", "")).strip().lower()
    table = str(primary.get("table", ""))
    trace: list[str] = []

    usable, reason = python_is_usable(python_row)
    primary_bool = tabfact_boolean(primary_answer)
    python_bool = tabfact_boolean(python_answer)
    optimized_dp_bool = tabfact_boolean(optimized_dp_answer)
    semantic_route = tabfact_semantic_route(question, table)
    if (
        usable
        and semantic_route
        and python_bool is not None
        and python_bool == optimized_dp_bool
        and python_bool != primary_bool
    ):
        trace.extend(
            [reason, f"routed_{semantic_route}", "optimized_python_dp_agreement"]
        )
        return (
            python_answer,
            "optimized_semantic_consensus",
            bool(python_row.get("correct", False)),
            trace,
        )

    if not blind.get("override_accepted", False):
        trace.append("blind_verifier_kept_primary")
        return primary_answer, "dp_baseline", bool(primary.get("correct", False)), trace

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
