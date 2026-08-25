"""Gold-blind double-verifier fusion for aligned TableQA result streams."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluation.metrics import summarize_records
from paper1_runtime.adapters import get_adapter
from semantic_audit.api_client import ParateraClient


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return str(value)


def norm(adapter, value: Any) -> str:
    value = adapter.normalize_prediction(value)
    if isinstance(value, (list, tuple)):
        return " | ".join(str(item).strip().lower() for item in value)
    if value is None:
        return ""
    text = re.sub(
        r"\s+",
        " ",
        str(value).strip().lower().replace("\u00a0", " ").replace("–", "-").replace("—", "-").replace("−", "-"),
    )
    numeric = text.replace(",", "")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric):
        try:
            return f"number:{float(numeric):.12g}"
        except ValueError:
            pass
    return text


def candidate_supports(adapter, dataset: str, question: str, candidate: Any, proposed: Any) -> bool:
    if norm(adapter, candidate) == norm(adapter, proposed):
        return True
    if dataset == "wtq" and re.search(r"\bhow long\b|\bhow many years\b", question.lower()):
        def first_number(value: Any) -> str:
            match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
            return f"{float(match.group(0)):.12g}" if match else ""
        left, right = first_number(candidate), first_number(proposed)
        return bool(left and left == right)
    return False


def invalid(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = re.sub(r"(?i)^final\s+answer\s*:\s*", "", value).strip().lower()
        return not text or text in {"none", "null", "nan", "n/a", "error", "unknown"}
    return False


def parse_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {}


def verifier_prompt(
    dataset: str,
    table: str,
    title: str,
    question: str,
    candidates: list[tuple[str, Any]],
    perspective: str,
    hide_candidates: bool = False,
) -> str:
    if dataset == "tabfact":
        rule = (
            "Return the JSON boolean true or false. Decompose the statement into atomic clauses; "
            "one false clause makes the whole conjunction false. Preserve negation, quantifiers, "
            "comparators, entity identity, and row scope."
        )
    else:
        rule = (
            "Return a string, number, or JSON list. For counts, first enumerate every matching row "
            "internally and report matched_row_count. For durations, distinguish an interval label "
            "from its length. For missing-year questions, compute the complement of the displayed years."
        )
    if perspective == "recompute":
        method = (
            "Recompute from scratch: identify target column, filters, operations, answer type, "
            "and cardinality before comparing candidates."
        )
    else:
        method = (
            "Act as a counterexample auditor: try to falsify each candidate against exact table rows, "
            "then independently derive the surviving answer."
        )
    candidate_text = (
        "Candidate answers are intentionally hidden; solve independently."
        if hide_candidates
        else json.dumps(
            [{"label": label, "answer": safe(answer)} for label, answer in candidates],
            ensure_ascii=False,
        )
    )
    return f"""Independently resolve a disputed TableQA item using only the supplied table.
Candidate labels and methods are not evidence. {method} {rule}
Return one JSON object only:
{{"answer": "or JSON-native value", "confidence": 0.0,
  "matched_row_count": null, "operations": ["..."],
  "evidence": [{{"row": "...", "column": "...", "value": "..."}}],
  "reason": "concise table-grounded derivation"}}
Use at most 6 evidence entries. Do not assume a hidden reference answer.
DATASET: {dataset}
TITLE: {title}
TABLE:
{table}
QUESTION: {question}
CANDIDATES: {candidate_text}
"""


def check_alignment(sources: list[list[dict[str, Any]]]) -> None:
    if len({len(rows) for rows in sources}) != 1:
        raise ValueError("source row counts differ")
    ids = [[row["question_id"] for row in rows] for rows in sources]
    if any(current != ids[0] for current in ids[1:]):
        raise ValueError("sources are not aligned by question_id")


def run(args: argparse.Namespace) -> int:
    if len(args.candidate) != len(args.label):
        raise ValueError("--candidate and --label counts must match")
    primary_rows = read_rows(Path(args.primary))
    candidate_rows = [read_rows(Path(path)) for path in args.candidate]
    check_alignment([primary_rows, *candidate_rows])
    adapter = get_adapter(args.dataset, sub_sample=False)
    output = HERE / "outputs" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    (output / "log").mkdir(exist_ok=True)
    result_path = output / "result.jsonl"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(result_path)
    if result_path.exists():
        result_path.unlink()
    client = ParateraClient(cache_dir=str(output / "cache"), timeout=args.timeout)
    config = {
        **vars(args),
        "gold_excluded_from_verifier_prompt": True,
        "acceptance": (
            "two distinct verifier prompts agree; both confidence thresholds pass; both provide "
            "evidence; proposed answer is supported by required number of non-primary candidates"
        ),
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    for index, primary in enumerate(primary_rows):
        started = time.perf_counter()
        primary_answer = primary.get("final_answer")
        alternatives = [rows[index].get("final_answer") for rows in candidate_rows]
        primary_norm = norm(adapter, primary_answer)
        alternative_norms = [norm(adapter, value) for value in alternatives]
        disagreement = any(value and value != primary_norm for value in alternative_norms)
        should_verify = disagreement or invalid(primary_answer)
        reviews: list[dict[str, Any]] = []
        raw_outputs: list[str] = []
        calls = 0
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if should_verify:
            labeled = [(args.primary_label, primary_answer), *list(zip(args.label, alternatives))]
            for perspective in ("recompute", "counterexample"):
                response = client.chat(
                    [{"role": "user", "content": verifier_prompt(
                        args.dataset,
                        primary["table"],
                        primary.get("title", ""),
                        primary["question"],
                        labeled,
                        perspective,
                        args.hide_candidates,
                    )}],
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                )
                raw = response.get("text", "") or ""
                parsed = parse_object(raw)
                try:
                    confidence = float(parsed.get("confidence", 0) or 0)
                except (TypeError, ValueError):
                    confidence = 0.0
                review = {
                    "perspective": perspective,
                    "answer": safe(parsed.get("answer")),
                    "confidence": confidence,
                    "evidence": safe(parsed.get("evidence") or []),
                    "operations": safe(parsed.get("operations") or []),
                    "matched_row_count": safe(parsed.get("matched_row_count")),
                    "reason": str(parsed.get("reason", "")),
                    "parse_success": bool(parsed and "answer" in parsed),
                }
                reviews.append(review)
                raw_outputs.append(raw)
                calls += int(response.get("api_calls", 0) or 0)
                response_usage = response.get("usage") or {}
                for key in usage:
                    usage[key] += int(response_usage.get(key, 0) or 0)

        review_norms = [norm(adapter, review.get("answer")) for review in reviews]
        verifier_agreement = bool(
            len(review_norms) == 2 and review_norms[0] and review_norms[0] == review_norms[1]
        )
        proposed = reviews[0].get("answer") if verifier_agreement else None
        proposed_norm = norm(adapter, proposed)
        support_count = sum(
            candidate_supports(adapter, args.dataset, primary["question"], candidate, proposed)
            for candidate in alternatives
            if norm(adapter, candidate)
        )
        confidence_pass = bool(
            reviews and all(review.get("confidence", 0) >= args.confidence_threshold for review in reviews)
        )
        evidence_pass = bool(reviews and all(review.get("evidence") for review in reviews))
        accepted = bool(
            verifier_agreement
            and confidence_pass
            and evidence_pass
            and support_count >= args.required_candidate_support
            and proposed_norm != primary_norm
        )
        final_answer = proposed if accepted else primary_answer
        mode = "verified_override" if accepted else "verified_keep_primary" if should_verify else "agreement_keep_primary"
        correct = adapter.is_correct(final_answer, primary["answer"])
        record = {
            "idx": primary["idx"],
            "answer": primary["answer"],
            "text": "\n---\n".join(raw_outputs),
            "transpose": primary.get("transpose", False),
            "resort": primary.get("resort", []),
            "question_id": primary["question_id"],
            "table_id": primary.get("table_id"),
            "title": primary.get("title", ""),
            "table": primary["table"],
            "question": primary["question"],
            "sample_id": primary.get("sample_id", primary["question_id"]),
            "method": "double_verifier_fusion",
            "primary_label": args.primary_label,
            "primary_answer": safe(primary_answer),
            "candidate_labels": args.label,
            "candidate_answers": safe(alternatives),
            "reviews": safe(reviews),
            "verifier_agreement": verifier_agreement,
            "candidate_support_count": support_count,
            "confidence_pass": confidence_pass,
            "evidence_pass": evidence_pass,
            "override_accepted": accepted,
            "selection_mode": mode,
            "selected_method": "verifier" if accepted else args.primary_label,
            "final_answer": safe(final_answer),
            "normalized_prediction": safe(adapter.normalize_prediction(final_answer)),
            "correct": bool(correct),
            "model": client.model,
            "api_usage": usage,
            "api_calls": calls,
            "latency_seconds": time.perf_counter() - started,
            "gold_excluded_from_verifier_prompt": True,
        }
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        (output / "log" / f"{primary['idx']}.json").write_text(
            json.dumps({"question_id": primary["question_id"], "reviews": reviews, "accepted": accepted, "final_answer": safe(final_answer), "correct": bool(correct)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        records.append(record)
        print(primary["idx"], primary["question_id"], mode, "support=", support_count, "correct=", bool(correct), flush=True)

    (output / "summary.json").write_text(
        json.dumps(summarize_records(records), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wtq", "tabfact"], required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--primary-label", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--required-candidate-support", type=int, default=1)
    parser.add_argument("--hide-candidates", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--overwrite", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
