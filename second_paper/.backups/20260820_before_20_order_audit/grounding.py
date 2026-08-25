"""Table-grounded normalization and evidence helpers."""
from __future__ import annotations
import math, re, unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, Optional
import pandas as pd

def normalize_text(value: Any) -> str:
    text=unicodedata.normalize("NFKC", str(value)).strip().lower()
    text=text.replace("–", "-").replace("—", "-").replace("−", "-")
    text=re.sub(r"\s+"," ",text)
    return text

def normalize_number(value: Any) -> Optional[float]:
    if isinstance(value,(int,float)) and not isinstance(value,bool): return float(value)
    text=normalize_text(value).replace(",","")
    try: return float(text)
    except (ValueError,TypeError): return None

def values_equal(a: Any,b: Any) -> bool:
    na,nb=normalize_number(a),normalize_number(b)
    if na is not None and nb is not None: return math.isclose(na,nb,rel_tol=1e-9,abs_tol=1e-9)
    return normalize_text(a)==normalize_text(b)

def closest_column(column: str, columns: Iterable[Any]) -> Optional[str]:
    names=[str(x) for x in columns]
    if column in names: return column
    target=normalize_text(column)
    for name in names:
        if normalize_text(name)==target: return name
    scored=[(SequenceMatcher(None,target,normalize_text(name)).ratio(),name) for name in names]
    return max(scored)[1] if scored and max(scored)[0]>=0.72 else None

def table_grounding(df: pd.DataFrame, used_columns: Iterable[str], filters: Iterable[Any]) -> Dict[str,Any]:
    columns=[str(x) for x in df.columns]; evidence=[]; unknown=[]; aliases={}
    for col in used_columns:
        actual=closest_column(str(col),columns)
        if actual is None: unknown.append(col); evidence.append(f"unknown_column:{col}")
        elif actual!=col: aliases[col]=actual; evidence.append(f"column_alias:{col}->{actual}")
    entities=[]; missing_entities=[]
    for spec in filters:
        col=aliases.get(spec.column,spec.column)
        if col in df.columns:
            found=any(values_equal(x,spec.value) for x in df[col].tolist())
            (entities if found else missing_entities).append({"column":col,"value":spec.value})
            evidence.append(f"cell_{'found' if found else 'missing'}:{col}={spec.value}")
    return {"columns":columns,"unknown_columns":unknown,"aliases":aliases,"entities_found":entities,"entities_missing":missing_entities,"evidence":evidence}
