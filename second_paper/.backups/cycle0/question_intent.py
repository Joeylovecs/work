"""Question Intent IR extraction with a deterministic fallback."""
from __future__ import annotations
import json, re
from typing import Any, Dict, Optional
import pandas as pd
from .intent_schema import FilterSpec, QuestionIntent, question_from_dict

def parse_json_object(text: str) -> Optional[Dict[str,Any]]:
    if not text: return None
    candidates=[text.strip()]
    candidates += re.findall(r"\{.*\}",text,re.S)
    for raw in candidates:
        try:
            value=json.loads(raw)
            if isinstance(value,dict): return value
        except Exception: pass
    return None

def heuristic_intent(question: str, df: Optional[pd.DataFrame]=None, dataset: str="wtq") -> QuestionIntent:
    q=question.lower(); cols=[str(c) for c in df.columns] if df is not None else []
    answer_type="boolean" if dataset.lower() in {"tabfact","tabularfact"} else ("number" if any(x in q for x in ["how many","how much","number of","total","average","mean","sum","difference","amount","year","date","long did","how old"]) else "entity")
    if any(x in q for x in ["which ","what country","what team","who ","what was the name"]): answer_type="entity"
    cardinality="multiple" if any(x in q for x in ["which countries","which teams","what are","list ","all "]) else "single"
    task="boolean_verification" if answer_type=="boolean" else "lookup"
    if any(x in q for x in ["total","sum","average","mean","how many","number of"]): task="aggregation"
    if any(x in q for x in ["highest","lowest","most","least","largest","smallest","top"]): task="ranking"
    if any(x in q for x in ["difference","more than","less than"]): task="comparison"
    agg=None
    if "average" in q or "mean" in q: agg="mean"
    elif "total" in q or "sum" in q: agg="sum"
    elif "how many" in q or "number of" in q: agg="count"
    targets=[]
    for col in cols:
        tokens=re.findall(r"[a-z0-9]+",col.lower())
        if tokens and any(t in q for t in tokens) and col.lower() not in {"year","date"}: targets.append(col)
    if not targets and cols and agg:
        numeric=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])] if df is not None else []
        if len(numeric)==1: targets=numeric
    filters=[]; operations=[]
    for col in cols:
        c=re.escape(col)
        patterns=[(rf"{c}\s*(?:is|=|equals?)\s*['\"]?([^,'\"?]+)","="),
                  (rf"{c}\s*(?:after|later than)\s*(\d{{4}})",">"),
                  (rf"{c}\s*(?:before|earlier than)\s*(\d{{4}})","<")]
        for pat,op in patterns:
            m=re.search(pat,q)
            if m: filters.append(FilterSpec(col,op,m.group(1).strip())); operations.append("filter")
    if "after " in q or "later than " in q or "before " in q or "earlier than " in q:
        for col in cols:
            if col.lower() in {"year","date"} or "year" in col.lower() or "date" in col.lower():
                m=re.search(r"\b(after|later than|before|earlier than)\s+(\d{4})",q)
                if m:
                    filters.append(FilterSpec(col,">" if m.group(1) in {"after","later than"} else "<",int(m.group(2)))); operations.append("filter"); break
    if filters and "filter" not in operations: operations.append("filter")
    if agg: operations.append("aggregate")
    ranking=None
    if task=="ranking":
        ranking={"direction":"asc" if any(x in q for x in ["lowest","least","smallest"]) else "desc","column":targets[0] if targets else None}
        operations.append("ranking")
    if answer_type=="boolean": operations.append("boolean_reduce")
    return QuestionIntent(answer_type=answer_type,cardinality=cardinality,task_type=task,target_columns=targets,filters=filters,operations=operations,aggregation=agg,ranking=ranking,boolean_polarity="true" if not any(x in q for x in ["not","never","no "]) else "false",evidence=["heuristic_question_parser"],confidence=0.35,source="heuristic")

def intent_from_model(text: str, question: str, df: Optional[pd.DataFrame]=None, dataset: str="wtq") -> QuestionIntent:
    value=parse_json_object(text)
    if value is not None:
        try:
            result=question_from_dict(value); result.source="llm"; result.confidence=max(result.confidence,0.7); return result
        except Exception: pass
    return heuristic_intent(question,df,dataset)
