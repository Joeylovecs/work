"""Joint inference over two independently generated result streams."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from second_paper.evaluation.metrics import summarize_records
from second_paper.paper1_runtime.adapters import get_adapter
from second_paper.paper1_runtime.prompts import joint_reasoning_prompt
from second_paper.semantic_audit.api_client import ParateraClient


def safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return str(value)


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def extract_answer(value: Any, dataset: str) -> Any:
    if value is None or isinstance(value, (list, int, float, bool)):
        return value
    text = str(value).strip()
    match = re.findall(r"Final Answer\s*:\s*(.+)", text, flags=re.I)
    if match:
        text = match[-1].strip()
    if dataset == "tabfact":
        low = text.lower()
        if "yes" in low or "true" in low:
            return "yes"
        if "no" in low or "false" in low:
            return "no"
    return text


def normalized(adapter, value: Any) -> str:
    value = adapter.normalize_prediction(value)
    if isinstance(value, (list, tuple)):
        return " | ".join(str(x).strip().lower() for x in value)
    return "" if value is None else str(value).strip().lower()


def compact_evidence(record: dict[str, Any]) -> str:
    payload = {
        "method": record.get("method"),
        "execution_success": record.get("final_execution_success"),
        "execution_result": record.get("final_execution_result"),
        "execution_warnings": record.get("execution_warnings"),
        "question_intent": record.get("question_intent_ir"),
        "code_intent": record.get("code_intent_ir"),
        "code_audit_status": record.get("audit_status"),
        "dp_audit_status": record.get("dp_audit_status"),
        "dp_review_trace": record.get("dp_review_trace"),
    }
    text = json.dumps(safe(payload), ensure_ascii=False)
    return text[:6500]


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> int:
    source_a = Path(args.source_a)
    source_b = Path(args.source_b)
    rows_a, rows_b = read_rows(source_a), read_rows(source_b)
    if len(rows_a) != len(rows_b):
        raise ValueError("Joint sources have different row counts")
    ids_a = [row["question_id"] for row in rows_a]
    ids_b = [row["question_id"] for row in rows_b]
    if ids_a != ids_b:
        raise ValueError("Joint sources are not aligned by question_id")

    adapter = get_adapter(args.dataset, sub_sample=False)
    output = HERE / "outputs" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    (output / "log").mkdir(exist_ok=True)
    result_path = output / "result.jsonl"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"{result_path} exists; use --overwrite")
    if result_path.exists():
        result_path.unlink()

    client = ParateraClient(cache_dir=str(output / "cache"), timeout=args.timeout)
    config = vars(args).copy()
    config.update({
        "source_a": str(source_a.resolve()),
        "source_b": str(source_b.resolve()),
        "question_ids": ids_a,
        "gold_excluded_from_joint_prompt": True,
    })
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []

    for left, right in zip(rows_a, rows_b):
        started = time.perf_counter()
        candidate_a = left.get("final_answer")
        candidate_b = right.get("final_answer")
        norm_a, norm_b = normalized(adapter, candidate_a), normalized(adapter, candidate_b)
        raw = ""
        calls = 0
        usage: dict[str, int] = {}
        selection: dict[str, Any]

        if norm_a and norm_a == norm_b:
            final_answer = candidate_a
            mode = "agreement"
            selection = {"choice": "A", "reason": "normalized candidates agree"}
        elif not norm_a and norm_b:
            final_answer = candidate_b
            mode = "candidate_b_only"
            selection = {"choice": "B", "reason": "candidate A is empty"}
        elif norm_a and not norm_b:
            final_answer = candidate_a
            mode = "candidate_a_only"
            selection = {"choice": "A", "reason": "candidate B is empty"}
        else:
            prompt = joint_reasoning_prompt(
                args.dataset,
                left["table"],
                left.get("title", ""),
                left["question"],
                str(candidate_a),
                str(candidate_b),
                compact_evidence(left),
                compact_evidence(right),
            )
            response = client.chat(
                [{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            raw = response.get("text", "") or ""
            calls = int(response.get("api_calls", 0) or 0)
            usage = response.get("usage") or {}
            selection = parse_json_object(raw)
            choice = str(selection.get("choice", "")).strip().upper()
            proposed = extract_answer(selection.get("answer"), args.dataset)
            if choice == "A":
                final_answer = candidate_a
            elif choice == "B":
                final_answer = candidate_b
            elif choice == "RECOMPUTE" and proposed is not None:
                final_answer = proposed
            elif proposed is not None:
                final_answer = proposed
                choice = "RECOMPUTE"
            else:
                final_answer = candidate_a if norm_a else candidate_b
                choice = "A" if norm_a else "B"
            selection["choice"] = choice
            mode = "llm_joint"

        correct = adapter.is_correct(final_answer, left["answer"])
        record = {
            "idx": left["idx"],
            "answer": left["answer"],
            "text": raw,
            "transpose": left.get("transpose", False),
            "resort": left.get("resort", []),
            "question_id": left["question_id"],
            "table_id": left.get("table_id"),
            "title": left.get("title", ""),
            "table": left["table"],
            "question": left["question"],
            "sample_id": left.get("sample_id", left["question_id"]),
            "method": "joint_from_results",
            "selected_method": selection.get("choice"),
            "selection_mode": mode,
            "source_a_label": args.label_a,
            "source_b_label": args.label_b,
            "candidate_a": safe(candidate_a),
            "candidate_b": safe(candidate_b),
            "candidate_a_correct": bool(adapter.is_correct(candidate_a, left["answer"])),
            "candidate_b_correct": bool(adapter.is_correct(candidate_b, left["answer"])),
            "source_agreement": bool(norm_a and norm_a == norm_b),
            "selection": safe(selection),
            "final_answer": safe(final_answer),
            "normalized_prediction": safe(adapter.normalize_prediction(final_answer)),
            "correct": bool(correct),
            "model": client.model,
            "api_usage": safe(usage),
            "api_calls": calls,
            "latency_seconds": time.perf_counter() - started,
            "gold_excluded_from_joint_prompt": True,
        }
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        (output / "log" / f"{left['idx']}.txt").write_text(
            json.dumps({
                "question_id": left["question_id"],
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "selection": selection,
                "final_answer": final_answer,
                "correct": correct,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        records.append(record)
        print(left["idx"], left["question_id"], args.label_a, "+", args.label_b, "correct=", correct, "mode=", mode, flush=True)

    (output / "summary.json").write_text(
        json.dumps(summarize_records(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wtq", "tabfact"], required=True)
    parser.add_argument("--source-a", required=True)
    parser.add_argument("--source-b", required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
