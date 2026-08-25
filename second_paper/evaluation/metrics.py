"""Auditor and paired-method diagnostics."""
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, Iterable

def _errors(record,level):
    block=record.get(level) or {}; return block.get("errors",[]) if isinstance(block,dict) else []

def summarize_records(records: Iterable[Dict[str,Any]]) -> Dict[str,Any]:
    rows=list(records); n=len(rows); exec_ok=sum(bool(r.get("execution_success")) for r in rows); correct=sum(bool(r.get("correct")) for r in rows)
    silent=sum(bool(r.get("execution_success")) and not bool(r.get("correct")) for r in rows)
    semantic=sum(bool(r.get("semantic_exception")) for r in rows)
    layers=Counter(); types=Counter()
    for r in rows:
        for level,key in (("GLOBAL","global_audit"),("OPERATOR","operator_audit"),("PARAMETER","parameter_audit")):
            for e in _errors(r,key): layers[level]+=1; types[e.get("error_type","unknown")]+=1
    return {"count":n,"accuracy":correct/n if n else 0.0,"execution_success_rate":exec_ok/n if n else 0.0,"execution_success_but_wrong":silent,"semantic_exception_count":semantic,"layer_error_counts":dict(layers),"error_type_counts":dict(types),"api_calls":sum(r.get("api_calls",0) or 0 for r in rows),"total_tokens":sum((r.get("api_usage") or {}).get("total_tokens",0) or 0 for r in rows),"avg_latency_seconds":sum(r.get("latency_seconds",0) or 0 for r in rows)/n if n else 0.0}

def compare(baseline, audited):
    def rid(r): return r.get("sample_id") or r.get("question_id") or r.get("idx")
    b={rid(r):r for r in baseline}; a={rid(r):r for r in audited}; ids=sorted(set(b)&set(a), key=str)
    silent=[sid for sid in ids if b[sid].get("execution_success") and not b[sid].get("correct")]
    detected=[sid for sid in silent if a[sid].get("semantic_exception")]
    repaired=[sid for sid in silent if a[sid].get("correct")]
    false_positive=[sid for sid in ids if b[sid].get("correct") and a[sid].get("semantic_exception")]
    degradation=[sid for sid in ids if b[sid].get("correct") and not a[sid].get("correct")]
    return {"paired_count":len(ids),"baseline":summarize_records([b[x] for x in ids]),"audited":summarize_records([a[x] for x in ids]),"silent_failure_candidates":silent,"semantic_failures_detected":detected,"detection_recall":len(detected)/len(silent) if silent else None,"repair_success":repaired,"false_positives":false_positive,"repair_degradation":degradation}
