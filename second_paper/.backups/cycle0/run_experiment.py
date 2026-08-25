"""Run paired baseline/audited experiments on paper-1 ordered samples."""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from second_paper.evaluation.paper1_compat import iter_samples, sample_ids, is_correct, normalize_prediction
from second_paper.semantic_audit.api_client import APIConfigError, ParateraClient
from second_paper.semantic_audit.ast_analyzer import analyze_code
from second_paper.semantic_audit.auditor import AuditConfig, audit_intent
from second_paper.semantic_audit.intent_schema import dumps
from second_paper.semantic_audit.llm_audit import llm_audit
from second_paper.semantic_audit.prompts import PROMPT_VERSION, code_prompt, intent_prompt, repair_prompt
from second_paper.semantic_audit.question_intent import heuristic_intent, intent_from_model
from second_paper.semantic_audit.runtime_trace import execute_code

def extract_code(text: str) -> str:
    if not text: return ""
    blocks=re.findall(r"```(?:python|py)?\s*(.*?)```",text,flags=re.I|re.S)
    if blocks: return blocks[-1].strip()
    text=text.strip()
    if "Action Input:" in text: text=text.split("Action Input:")[-1]
    return text.strip().strip('`')

def json_safe(value):
    if value is None or isinstance(value,(str,int,float,bool)): return value
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    return str(value)

def call(client,prompt,temperature=0.0,max_tokens=1200):
    data=client.chat([{"role":"user","content":prompt}],temperature=temperature,max_tokens=max_tokens)
    return data.get("text","") or "", data

def audit_call(client,args,dataset,table,question,q_ir,code,code_ir,df):
    if args.audit_mode=="llm_only": return llm_audit(client,dataset,table,question,q_ir,code,code_ir)
    return audit_intent(q_ir,code_ir,df,AuditConfig(level=args.audit_level,mode=args.audit_mode)), {"api_calls":0,"usage":{}}

def add_usage(total_usage,data):
    usage=data.get("usage",{}) or {}
    return {k:(total_usage.get(k,0) or 0)+(usage.get(k,0) or 0) for k in ("prompt_tokens","completion_tokens","total_tokens")}

def run(args):
    dataset=args.dataset.lower(); exp=ROOT/"second_paper"/"outputs"/args.experiment
    exp.mkdir(parents=True,exist_ok=True); (exp/"logs").mkdir(exist_ok=True); (exp/"cases").mkdir(exist_ok=True)
    selected_path=Path(args.selected_file) if args.selected_file else exp/"selected_samples.json"
    if selected_path.exists(): selected=json.loads(selected_path.read_text(encoding="utf-8"))["sample_ids"]
    else:
        selected=sample_ids(dataset,args.limit); selected_path.parent.mkdir(parents=True,exist_ok=True); selected_path.write_text(json.dumps({"dataset":dataset,"sample_ids":selected,"selection":"paper1_order_fixed_slice","debug_test_slice":True},ensure_ascii=False,indent=2),encoding="utf-8")
    config={"dataset":dataset,"method":args.method,"audit_level":args.audit_level,"audit_mode":args.audit_mode,"model":args.model,"prompt_version":PROMPT_VERSION,"temperature":args.temperature,"max_semantic_repairs":args.max_repairs,"selected_samples":selected,"debug_test_slice":True}
    (exp/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    try: client=ParateraClient(cache_dir=str(exp/"cache"),timeout=args.timeout)
    except APIConfigError as exc:
        print(str(exc),file=sys.stderr); return 2
    out_path=exp/"predictions.jsonl"
    if out_path.exists() and not args.overwrite: out_path.unlink()
    records=[]
    for i,s in enumerate(iter_samples(dataset,selected_ids=selected)):
        started=time.perf_counter(); total_usage={}; total_calls=0; raw_outputs=[]; q_ir=None; code=""; code_response={}; execution={}; audit=None; initial_audit=None; initial_execution=None; initial_code_ir=None; initial_prediction=None; initial_correct=False; repaired_code=None; repair_count=0; repair_trace=[]
        q_prompt=intent_prompt(dataset,s.table_md,s.question)
        if args.method=="audit":
            q_text,q_response=call(client,q_prompt,args.temperature,1000); raw_outputs.append(q_text); q_ir=intent_from_model(q_text,s.question,s.df,dataset); total_usage=q_response.get("usage",{}) or {}; total_calls+=q_response.get("api_calls",0)
        code_text,code_response=call(client,code_prompt(dataset,s.table_md,s.question),args.temperature,1600); raw_outputs.append(code_text); total_calls+=code_response.get("api_calls",0); total_usage={k:(total_usage.get(k,0) or 0)+(code_response.get("usage",{}) or {}).get(k,0) for k in ("prompt_tokens","completion_tokens","total_tokens")}; code=extract_code(code_text)
        execution=execute_code(code,s.df); code_ir=analyze_code(code,execution) if execution.get("success") else analyze_code(code,{})
        if args.method=="audit" and q_ir is None: q_ir=heuristic_intent(s.question,s.df,dataset)
        if args.method=="audit" and execution.get("success"):
            audit,audit_response=audit_call(client,args,dataset,s.table_md,s.question,q_ir,code,code_ir,s.df); total_calls+=audit_response.get("api_calls",0); total_usage=add_usage(total_usage,audit_response)
        initial_execution=execution
        initial_code_ir=code_ir
        initial_audit=audit
        initial_prediction=execution.get("result") if execution.get("success") else None
        initial_correct=is_correct(dataset,initial_prediction,s.gold)
        if args.method=="audit" and execution.get("success") and audit is not None:
            while audit.semantic_exception and repair_count<args.max_repairs:
                diagnostic=dumps(audit.to_dict()); rtext,rdata=call(client,repair_prompt(dataset,s.table_md,s.question,code,diagnostic),args.temperature,1600); raw_outputs.append(rtext); total_calls+=rdata.get("api_calls",0); total_usage={k:(total_usage.get(k,0) or 0)+(rdata.get("usage",{}) or {}).get(k,0) for k in ("prompt_tokens","completion_tokens","total_tokens")}; repaired_code=extract_code(rtext); repair_count+=1; rex=execute_code(repaired_code,s.df); repair_trace.append({"repair_count":repair_count,"diagnostic":audit.to_dict(),"repaired_code":repaired_code,"execution":rex}); code=repaired_code; execution=rex; code_ir=analyze_code(code,execution) if execution.get("success") else analyze_code(code,{});
                audit_response={"api_calls":0,"usage":{}}
                if execution.get("success"):
                    audit,audit_response=audit_call(client,args,dataset,s.table_md,s.question,q_ir,code,code_ir,s.df)
                    total_calls+=audit_response.get("api_calls",0); total_usage=add_usage(total_usage,audit_response)
                else:
                    audit=None
        prediction=execution.get("result") if execution.get("success") else None
        final_correct=is_correct(dataset,prediction,s.gold)
        record={**s.metadata(),"method":args.method,"model":client.model,"prompt_version":PROMPT_VERSION,"raw_model_output":"\n---\n".join(raw_outputs),"generated_python":extract_code(code_text),"execution_success":bool(initial_execution.get("success")),"execution_error":initial_execution.get("error"),"execution_result":json_safe(initial_execution.get("observed_value")),"execution_evidence":json_safe(initial_execution.get("evidence",{})),"initial_execution_success":bool(initial_execution.get("success")),"initial_execution_error":initial_execution.get("error"),"initial_execution_result":json_safe(initial_execution.get("observed_value")),"final_execution_success":bool(execution.get("success")),"final_execution_error":execution.get("error"),"final_execution_result":json_safe(execution.get("observed_value")),"question_intent_ir":q_ir.to_dict() if q_ir else None,"code_intent_ir":initial_code_ir.to_dict(),"global_audit":initial_audit.global_audit.to_dict() if initial_audit else None,"operator_audit":initial_audit.operator_audit.to_dict() if initial_audit else None,"parameter_audit":initial_audit.parameter_audit.to_dict() if initial_audit else None,"semantic_exception":initial_audit.to_dict() if initial_audit else None,"final_audit":audit.to_dict() if audit else None,"repair_hint":initial_audit.repair_hint if initial_audit else "","repaired_code":repaired_code,"repair_trace":json_safe(repair_trace),"repair_count":repair_count,"initial_answer":json_safe(initial_prediction),"initial_correct":bool(initial_correct),"final_answer":json_safe(prediction),"normalized_prediction":json_safe(normalize_prediction(dataset,prediction)),"correct":bool(final_correct),"api_usage":total_usage,"api_calls":total_calls,"latency_seconds":time.perf_counter()-started}
        with out_path.open("a",encoding="utf-8") as f: f.write(json.dumps(json_safe(record),ensure_ascii=False)+"\n")
        records.append(record); print(i,s.sample_id,"correct="+str(final_correct),"exec="+str(execution.get("success")),flush=True)
    from second_paper.evaluation.metrics import summarize_records
    summary=summarize_records(records); (exp/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); return 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",choices=["wtq","tabfact"],required=True); ap.add_argument("--method",choices=["baseline","audit"],required=True); ap.add_argument("--experiment",required=True); ap.add_argument("--limit",type=int,default=5); ap.add_argument("--selected-file"); ap.add_argument("--audit-level",choices=["global","global_operator","full"],default="full"); ap.add_argument("--audit-mode",choices=["hybrid","llm_only"],default="hybrid"); ap.add_argument("--model",default="DeepSeek-V3.2"); ap.add_argument("--temperature",type=float,default=0.0); ap.add_argument("--max-repairs",type=int,default=2); ap.add_argument("--timeout",type=float,default=120.0); ap.add_argument("--overwrite",action="store_true"); args=ap.parse_args(); raise SystemExit(run(args))
if __name__=="__main__": main()
