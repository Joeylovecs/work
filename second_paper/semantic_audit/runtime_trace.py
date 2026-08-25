"""Deterministic execution wrapper with compact runtime evidence."""
from __future__ import annotations

import ast
import contextlib
import io
import time
import traceback
from typing import Any, Dict

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Series):
        return value.astype(object).tolist()[:100]
    if isinstance(value, pd.Index):
        return value.astype(object).tolist()[:100]
    if isinstance(value, np.ndarray):
        return value.tolist()[:100]
    if isinstance(value, (list, tuple)):
        return [_json_value(x) for x in value[:100]]
    return str(value)


def _type_name(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "number"
    if isinstance(value, (list, tuple, pd.Series, pd.Index, np.ndarray)):
        return "list"
    if isinstance(value, pd.DataFrame):
        return "table"
    if isinstance(value, str):
        return "string"
    return type(value).__name__ if value is not None else "null"


def _semantic_warnings(result: Any, frames: list[dict[str, Any]]) -> list[str]:
    warnings = []
    if result is None:
        warnings.append("null_final_answer")
    elif isinstance(result, pd.DataFrame):
        warnings.append("table_returned_instead_of_answer")
        if result.empty:
            warnings.append("empty_table_result")
    elif isinstance(result, (pd.Series, pd.Index, np.ndarray)):
        if len(result) == 0:
            warnings.append("empty_collection_result")
        else:
            warnings.append("collection_returned_instead_of_scalar_or_list")
    elif isinstance(result, (list, tuple)) and len(result) == 0:
        warnings.append("empty_collection_result")
    elif isinstance(result, (float, np.floating)) and np.isnan(result):
        warnings.append("nan_final_answer")
    elif isinstance(result, str) and not result.strip():
        warnings.append("empty_string_result")
    if any(frame.get("name") != "df" and frame.get("shape", [1])[0] == 0 for frame in frames):
        warnings.append("empty_intermediate_dataframe")
    if not frames:
        warnings.append("no_runtime_dataframe_evidence")
    return warnings


def execute_code(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    started = time.perf_counter()
    out = io.StringIO()
    memory: dict[str, Any] = {}
    result = None
    try:
        tree = ast.parse(code)
        glb = {"pd": pd, "np": np, "df": df.copy(deep=True), "__builtins__": __builtins__}
        loc: dict[str, Any] = {}
        body = list(tree.body)
        last = body.pop() if body and isinstance(body[-1], ast.Expr) else None
        with contextlib.redirect_stdout(out):
            if body:
                exec(compile(ast.Module(body=body, type_ignores=[]), "<generated>", "exec"), glb, loc)
            if last is not None:
                result = eval(compile(ast.Expression(last.value), "<generated>", "eval"), glb, loc)
        for key in ("answer", "result", "output", "prediction"):
            if key in loc:
                result = loc[key]
            elif key in glb:
                result = glb[key]

        namespace = {**glb, **loc}
        frames = []
        trace: list[dict[str, Any]] = []
        base_rows = int(len(glb["df"]))
        for name, value in namespace.items():
            if isinstance(value, pd.DataFrame):
                frame = {"name": name, "shape": [int(x) for x in value.shape], "columns": [str(x) for x in value.columns]}
                frames.append(frame)
                if name == "df":
                    trace.append({"operation": "input_table", "name": name, "rows": int(len(value)), "columns": frame["columns"]})
                elif len(value) < base_rows:
                    trace.append({"operation": "filter", "name": name, "rows_before": base_rows, "rows_after": int(len(value)), "columns": frame["columns"]})
                else:
                    trace.append({"operation": "dataframe_state", **frame})
            elif isinstance(value, pd.Series):
                trace.append({"operation": "series_state", "name": name, "length": int(len(value)), "column": str(value.name) if value.name is not None else None})
        trace.append({"operation": "return", "output_type": _type_name(result), "value": _json_value(result)})
        warnings = _semantic_warnings(result, frames)
        evidence = {
            "final_object_type": _type_name(result),
            "final_value": _json_value(result),
            "dataframes": frames,
            "operation_trace": trace,
            "semantic_warnings": warnings,
            "stdout": out.getvalue()[:4000],
            "latency_seconds": time.perf_counter() - started,
        }
        return {
            "success": True,
            "error": None,
            "result": result,
            "observed_output_type": _type_name(result),
            "observed_value": _json_value(result),
            "evidence": evidence,
            "memory": memory,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "result": None,
            "observed_output_type": "unknown",
            "observed_value": None,
            "evidence": {
                "stdout": out.getvalue()[:4000],
                "traceback": traceback.format_exc(limit=3),
                "operation_trace": [],
                "semantic_warnings": ["execution_exception"],
                "latency_seconds": time.perf_counter() - started,
            },
            "memory": memory,
        }
