"""Gold-blind structured direct-reasoning baseline for raw-order TableQA data."""
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
from second_paper.semantic_audit.api_client import ParateraClient


def prompt(dataset: str, table: str, title: str, question: str) -> str:
    answer_rule = (
        "Return the answer as the JSON boolean true or false. Verify the exact claim, "
        "including every entity, number, comparator, quantifier, and negation."
        if dataset == "tabfact"
        else
        "The answer may be a string, number, or JSON list. Preserve exact table text "
        "for lookup answers and return a list only for a genuinely plural question."
    )
    return f"""Solve this table question without Python and return one JSON object only.
Use the table as the sole source. Explicitly identify the target column, row filters,
comparison/aggregation/ranking/arithmetic operations, answer type, and cardinality.
Recompute the result rather than copying a nearby value. {answer_rule}
Required JSON schema:
{{"answer": "or JSON-native value", "answer_type": "entity|number|date|duration|boolean|list",
  "cardinality": "single|multiple", "filters": ["..."], "operations": ["..."],
  "evidence": [{{"row": "...", "column": "...", "value": "..."}}],
  "confidence": 0.0, "reason": "concise table-grounded derivation"}}
Do not assume a hidden reference answer. Do not output markdown fences or extra prose.
DATASET: {dataset}
TITLE: {title}
TABLE:
{table}
QUESTION: {question}
"""


def safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return str(value)


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


def fallback(text: str, dataset: str) -> Any:
    matches = re.findall(r"Final Answer\s*:\s*(.+)", text or "", flags=re.I)
    value = matches[-1].strip() if matches else None
    if value is not None and dataset == "tabfact":
        low = value.lower()
        if low in {"yes", "true", "1"}:
            return True
        if low in {"no", "false", "0"}:
            return False
    return value


def run(args: argparse.Namespace) -> int:
    adapter = get_adapter(args.dataset, sub_sample=False)
    samples = list(adapter.iter_range(args.start, args.end))
    if not samples:
        raise ValueError("requested interval contains no samples")
    output = HERE / "outputs" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    (output / "log").mkdir(exist_ok=True)
    result_path = output / "result.jsonl"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"{result_path} exists; use a new experiment or --overwrite")
    if result_path.exists():
        result_path.unlink()

    client = ParateraClient(cache_dir=str(output / "cache"), timeout=args.timeout)
    sample_ids = [sample.sample_id for sample in samples]
    config = {
        **vars(args),
        "sample_ids": sample_ids,
        "raw_dataset_order": True,
        "gold_excluded_from_prompt": True,
        "method": "structured_dp",
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "selected_samples.json").write_text(
        json.dumps({"dataset": args.dataset, "sample_ids": sample_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    for sample in samples:
        started = time.perf_counter()
        response = client.chat(
            [{"role": "user", "content": prompt(args.dataset, sample.table_md, sample.title, sample.question)}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        raw = response.get("text", "") or ""
        parsed = parse_object(raw)
        final_answer = parsed.get("answer") if "answer" in parsed else fallback(raw, args.dataset)
        correct = adapter.is_correct(final_answer, sample.gold)
        record = {
            **sample.paper1_metadata(),
            "sample_id": sample.sample_id,
            "method": "structured_dp",
            "selected_method": "structured_dp",
            "text": raw,
            "structured_reasoning": safe(parsed),
            "parse_success": bool(parsed and "answer" in parsed),
            "final_answer": safe(final_answer),
            "normalized_prediction": safe(adapter.normalize_prediction(final_answer)),
            "correct": bool(correct),
            "model": client.model,
            "api_usage": safe(response.get("usage") or {}),
            "api_calls": int(response.get("api_calls", 0) or 0),
            "latency_seconds": time.perf_counter() - started,
            "gold_excluded_from_prompt": True,
        }
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        (output / "log" / f"{sample.flat_index}.json").write_text(
            json.dumps(
                {
                    "question_id": sample.question_id,
                    "raw": raw,
                    "parsed": safe(parsed),
                    "final_answer": safe(final_answer),
                    "correct": bool(correct),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        records.append(record)
        print(
            sample.flat_index,
            sample.question_id,
            "structured_dp",
            "correct=",
            bool(correct),
            "parsed=",
            bool(parsed and "answer" in parsed),
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
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
