"""Question Intent IR extraction with deterministic table-grounded fallback."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import pandas as pd

from .grounding import closest_column, normalize_text
from .intent_schema import FilterSpec, QuestionIntent, question_from_dict


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidates = [text.strip()]
    candidates += re.findall(r"\{.*\}", text, re.S)
    for raw in candidates:
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return None


def _filter_value_is_anchored(filter_spec: FilterSpec, question: str) -> bool:
    q = re.sub(r"\s+", " ", question.lower()).strip()
    raw = normalize_text(filter_spec.value)
    if not raw:
        return False
    if raw in q:
        return True
    core = re.sub(r"\s*\([^)]*\)\s*$", "", raw).strip()
    if core and core in q:
        return True
    nums = re.findall(r"\d+(?:\.\d+)?", raw)
    return bool(nums) and all(num in q for num in nums)


def normalize_boolean_polarity(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return None
    value = str(value or "").strip().lower()
    if value in {"positive", "affirmative", "assertive"}:
        return "positive"
    if value in {"negative", "negated", "not", "explicit_negation"}:
        return "negative"
    return None


def _target_columns(question: str, columns: list[str]) -> list[str]:
    q = normalize_text(question)
    targets: list[str] = []

    def token_mentioned(token: str) -> bool:
        """Match a schema token and its ordinary English plural in the question."""
        forms = {token, f"{token}s", f"{token}es"}
        if token.endswith("y") and len(token) > 1:
            forms.add(f"{token[:-1]}ies")
        return any(re.search(rf"\b{re.escape(form)}\b", q) for form in forms)

    hint_groups = [
        (("how long", "duration", "time did", "finish"), ("time", "duration")),
        (("country", "nation", "nationality"), ("country", "nation", "nationality")),
        (("team", "club"), ("team", "club")),
        (("point", "score"), ("point", "score")),
        (("rank", "position", "standing"), ("rank", "position", "standing")),
        (("year", "date", "when"), ("year", "date")),
        (("who", "cyclist", "player", "name"), ("cyclist", "player", "name")),
    ]
    for col in columns:
        c = normalize_text(col)
        tokens = re.findall(r"[a-z0-9]+", c)
        direct = bool(tokens) and any(token_mentioned(token) for token in tokens if len(token) > 2)
        semantic = any(any(trigger in q for trigger in triggers) and any(hint in c for hint in hints) for triggers, hints in hint_groups)
        if direct or semantic:
            targets.append(col)
    return targets


def _entity_filters(question: str, df: Optional[pd.DataFrame]) -> list[FilterSpec]:
    if df is None or df.empty:
        return []
    q = normalize_text(question)
    found: list[tuple[int, FilterSpec]] = []
    stop = {"yes", "no", "true", "false", "none", "nan", "first", "second", "third"}
    for col in df.columns:
        series = df[col].dropna().astype(str)
        for raw in series.drop_duplicates().head(500):
            full = normalize_text(raw)
            if not full or full in stop:
                continue
            try:
                float(full.replace(",", ""))
                continue
            except ValueError:
                pass
            core = re.sub(r"\s*\([^)]*\)\s*$", "", full).strip()
            core = re.sub(r"\s+", " ", core)
            candidates = sorted({full, core}, key=len, reverse=True)
            matched = next((x for x in candidates if len(x) >= 4 and re.search(rf"(?<!\w){re.escape(x)}(?!\w)", q)), None)
            if matched:
                op = "=" if matched == full else "contains"
                found.append((len(matched), FilterSpec(str(col), op, matched)))
    found.sort(key=lambda item: item[0], reverse=True)
    result: list[FilterSpec] = []
    used_columns: set[str] = set()
    for _, spec in found:
        key = normalize_text(spec.column)
        if key not in used_columns:
            result.append(spec)
            used_columns.add(key)
    return result


def heuristic_intent(question: str, df: Optional[pd.DataFrame] = None, dataset: str = "wtq") -> QuestionIntent:
    q = normalize_text(question)
    columns = [str(c) for c in df.columns] if df is not None else []
    is_tabfact = dataset.lower() in {"tabfact", "tabularfact"}
    targets = _target_columns(question, columns)
    filters = _entity_filters(question, df)

    if is_tabfact:
        answer_type = "boolean"
    elif "how long" in q and any("time" in normalize_text(c) or "duration" in normalize_text(c) for c in columns):
        answer_type = "duration"
    elif any(x in q for x in ["what year", "which year", "what date", "when "]):
        answer_type = "date"
    elif any(x in q for x in ["how many", "how much", "number of", "total", "average", "mean", "sum", "difference", "amount", "how old"]):
        answer_type = "number"
    else:
        answer_type = "entity"

    cardinality = "multiple" if any(x in q for x in ["which countries", "which teams", "what are", "list ", " all ", "which other"]) else "single"
    task = "boolean_verification" if is_tabfact else "lookup"
    if any(x in q for x in ["total", "sum", "average", "mean", "how many", "number of"]):
        task = "aggregation"
    if any(x in q for x in ["highest", "lowest", "most", "least", "largest", "smallest", "top ", "first ", "next after"]):
        task = "ranking"
    if any(x in q for x in ["difference", "more than", "less than"]):
        task = "comparison"

    aggregation = None
    if "average" in q or "mean" in q:
        aggregation = "mean"
    elif "how many" in q or "number of" in q:
        aggregation = (
            "count_distinct"
            if any(term in q for term in ["different", "distinct", "unique", "different kinds"])
            else "count"
        )
    elif "total" in q or "sum" in q:
        aggregation = "sum"

    operations: list[str] = []
    if filters:
        operations.append("filter")
    if task == "ranking":
        operations.append("ranking")
    if aggregation:
        operations.append("aggregate")
    if targets:
        operations.append("select")
    if is_tabfact:
        operations.append("boolean_reduce")

    ranking = None
    if task == "ranking":
        ranking = {
            "direction": "asc" if any(x in q for x in ["lowest", "least", "smallest"]) else "desc",
            "column": targets[0] if targets else None,
        }

    grounded = bool(targets) and (bool(filters) or task != "lookup" or is_tabfact)
    confidence = 0.78 if grounded else 0.35
    evidence = ["heuristic_question_parser"]
    if targets:
        evidence.append("table_grounded_target:" + ",".join(targets))
    if filters:
        evidence.extend(f"question_entity:{f.column}:{f.value}" for f in filters)
    return QuestionIntent(
        answer_type=answer_type,
        cardinality=cardinality,
        task_type=task,
        target_columns=targets,
        filters=filters,
        operations=operations,
        operation_sequence=list(operations),
        aggregation=aggregation,
        ranking=ranking,
        boolean_polarity="negative" if re.search(r"\b(not|never|no)\b", q) else "positive",
        evidence=evidence,
        confidence=confidence,
        source="heuristic",
    )


def intent_from_model(text: str, question: str, df: Optional[pd.DataFrame] = None, dataset: str = "wtq") -> QuestionIntent:
    value = parse_json_object(text)
    if value is not None:
        try:
            result = question_from_dict(value)
            columns = [str(c) for c in df.columns] if df is not None else []
            normalized_targets = []
            for target in result.target_columns:
                actual = closest_column(str(target), columns) if columns else str(target)
                if actual and actual not in normalized_targets:
                    normalized_targets.append(actual)
            result.target_columns = normalized_targets
            normalized_filters = []
            for spec in result.filters:
                actual = closest_column(spec.column, columns) if columns else spec.column
                if actual and _filter_value_is_anchored(spec, question):
                    normalized_filters.append(FilterSpec(actual, spec.op, spec.value))
            result.filters = normalized_filters
            result.operations = list(dict.fromkeys(result.operations or []))
            result.operation_sequence = list(result.operation_sequence or result.operations)
            result.source = "llm"
            result.confidence = max(float(result.confidence or 0.0), 0.8)
            result.boolean_polarity = normalize_boolean_polarity(result.boolean_polarity)
            if not isinstance(result.ranking, dict):
                result.ranking = None
            result.evidence = list(result.evidence or []) + ["low_confidence_llm_fallback"]
            return result
        except Exception:
            pass
    return heuristic_intent(question, df, dataset)
