"""Table-grounded normalization and evidence helpers."""
from __future__ import annotations

import math
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text)


def normalize_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = normalize_text(value).replace(",", "")
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def values_equal(a: Any, b: Any) -> bool:
    na, nb = normalize_number(a), normalize_number(b)
    if na is not None and nb is not None:
        return math.isclose(na, nb, rel_tol=1e-9, abs_tol=1e-9)
    return normalize_text(a) == normalize_text(b)


def value_match_type(cell: Any, expected: Any, op: str = "=") -> Optional[str]:
    if values_equal(cell, expected):
        return "numeric_equal" if normalize_number(cell) is not None and normalize_number(expected) is not None else "normalized_exact"
    cell_text, expected_text = normalize_text(cell), normalize_text(expected)
    if op == "contains" and expected_text and expected_text in cell_text:
        return "normalized_substring"
    if expected_text and len(expected_text) >= 4 and expected_text in cell_text:
        return "grounded_alias"
    return None


def closest_column(column: str, columns: Iterable[Any]) -> Optional[str]:
    names = [str(x) for x in columns]
    if column in names:
        return column
    target = normalize_text(column)
    for name in names:
        if normalize_text(name) == target:
            return name
    scored = [(SequenceMatcher(None, target, normalize_text(name)).ratio(), name) for name in names]
    return max(scored)[1] if scored and max(scored)[0] >= 0.72 else None


def table_grounding(df: pd.DataFrame, used_columns: Iterable[str], filters: Iterable[Any]) -> Dict[str, Any]:
    columns = [str(x) for x in df.columns]
    evidence: list[dict[str, Any]] = []
    unknown: list[str] = []
    aliases: dict[str, str] = {}
    for col in used_columns:
        actual = closest_column(str(col), columns)
        if actual is None:
            unknown.append(str(col))
            evidence.append({"kind": "column", "requested": str(col), "matched": None, "passed": False})
        else:
            if actual != col:
                aliases[str(col)] = actual
            evidence.append({
                "kind": "column",
                "requested": str(col),
                "matched": actual,
                "match_type": "exact" if actual == col else "normalized_alias",
                "passed": True,
            })

    entities: list[dict[str, Any]] = []
    missing_entities: list[dict[str, Any]] = []
    for spec in filters:
        col = aliases.get(str(spec.column), str(spec.column))
        actual_col = closest_column(col, columns)
        item = {"column": actual_col or col, "value": spec.value, "op": spec.op}
        if actual_col is None:
            missing_entities.append(item)
            evidence.append({"kind": "entity", **item, "matched_cell": None, "passed": False})
            continue
        match = None
        for cell in df[actual_col].tolist():
            match_type = value_match_type(cell, spec.value, spec.op)
            if match_type:
                match = (cell, match_type)
                break
        if match:
            grounded = {**item, "matched_cell": match[0], "match_type": match[1]}
            entities.append(grounded)
            evidence.append({"kind": "entity", **grounded, "passed": True})
        else:
            missing_entities.append(item)
            evidence.append({"kind": "entity", **item, "matched_cell": None, "passed": False})
    return {
        "columns": columns,
        "unknown_columns": unknown,
        "aliases": aliases,
        "entities_found": entities,
        "entities_missing": missing_entities,
        "evidence": evidence,
    }
