# Optional LLM-only audit ablation.
from __future__ import annotations
import json, re
from .intent_schema import AuditResult, LevelAudit, SemanticError, dumps

def _obj(text):
    for raw in [text.strip()]+re.findall(r"\{.*\}",text,re.S):
        try:
            value=json.loads(raw)
            if isinstance(value,dict): return value
        except Exception: pass
    return None

def llm_audit(client, dataset, table, question, question_ir, code, code_ir):
    prompt=f"""Audit whether the executed Python code faithfully implements the table question. Return JSON only with keys passed, repair_hint, global, operator, parameter. Each level has passed and errors; each error has level, error_type, expected, actual, evidence, confidence, repair_hint. Audit semantic mismatch, not syntax.
DATASET: {dataset}
QUESTION: {question}
QUESTION_INTENT_IR: {dumps(question_ir)}
CODE: {code}
CODE_INTENT_IR: {dumps(code_ir)}
TABLE:
{table}
"""
    response=client.chat([{"role":"user","content":prompt}],temperature=0.0,max_tokens=1600)
    data=_obj(response.get("text","")); levels={}
    if data is None:
        e=SemanticError("GLOBAL","LLM_AUDIT_PARSE_ERROR",evidence=["invalid_json"],confidence=0.1,repair_hint="Return a structured semantic audit.")
        levels={"global":LevelAudit("GLOBAL",False,[e],0.1),"operator":LevelAudit("OPERATOR"),"parameter":LevelAudit("PARAMETER")}
    else:
        for key,name in (("global","GLOBAL"),("operator","OPERATOR"),("parameter","PARAMETER")):
            block=data.get(key) or {}; errors=[]
            for raw in block.get("errors",[]):
                raw=dict(raw); raw["level"]=raw.get("level",name); raw["evidence"]=raw.get("evidence",[]); raw["confidence"]=raw.get("confidence",0.7); raw["repair_hint"]=raw.get("repair_hint","")
                errors.append(SemanticError(**{k:raw.get(k) for k in ("level","error_type","expected","actual","evidence","confidence","repair_hint")}))
            levels[key]=LevelAudit(name,not errors,errors,min((e.confidence for e in errors),default=1.0))
    errors=[e for block in levels.values() for e in block.errors]
    result=AuditResult(passed=not errors,semantic_exception=bool(errors),global_audit=levels["global"],operator_audit=levels["operator"],parameter_audit=levels["parameter"],repair_hint=" ".join(e.repair_hint for e in errors)[:1500],confidence=min((e.confidence for e in errors),default=1.0),mode="llm_only")
    return result,response
