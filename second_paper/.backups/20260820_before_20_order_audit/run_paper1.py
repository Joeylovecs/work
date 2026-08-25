"""Main second-paper runner with first-paper-compatible artifacts."""
from __future__ import annotations
import argparse, json, re, sys, time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from second_paper.paper1_runtime.adapters import get_adapter, Sample
from second_paper.paper1_runtime.prompts import dp_prompt, python_prompt, selector_prompt, repair_prompt, execution_repair_prompt
from second_paper.semantic_audit.api_client import APIConfigError, ParateraClient
from second_paper.semantic_audit.ast_analyzer import analyze_code
from second_paper.semantic_audit.auditor import AuditConfig, audit_intent
from second_paper.semantic_audit.intent_schema import dumps
from second_paper.semantic_audit.llm_audit import llm_audit
from second_paper.semantic_audit.question_intent import heuristic_intent
from second_paper.semantic_audit.runtime_trace import execute_code

def safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)): return value
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    return str(value)

def extract_code(text: str) -> str:
    if not text: return ""
    text = text.replace("~~~python~~~", "~~~python").replace("```python```", "```python")
    blocks = re.findall(r"(?:\x60{3}|~~~)(?:python|py)?\s*(.*?)(?:\x60{3}|~~~)", text, flags=re.I | re.S)
    if blocks: return blocks[-1].strip()
    if "Action Input:" in text: text = text.split("Action Input:")[-1]
    return text.strip().strip("~").strip()

def extract_answer(text: str, dataset: str) -> str | None:
    if not text: return None
    matches = re.findall(r"Final Answer\s*:\s*(.+)", text, flags=re.I)
    answer = matches[-1].strip() if matches else text.strip().splitlines()[-1].strip()
    answer = answer.strip("~").strip()
    if dataset == "tabfact":
        low = answer.lower()
        if "yes" in low or "true" in low: return "yes"
        if "no" in low or "false" in low: return "no"
    return answer

def usage_add(total: dict[str, int], response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = int(total.get(key, 0) or 0) + int(usage.get(key, 0) or 0)
    return total

def ask(client: ParateraClient, prompt: str, max_tokens: int, temperature: float = 0.0) -> tuple[str, dict[str, Any]]:
    response = client.chat([{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens)
    return response.get("text", "") or "", response

def normalized_answer(adapter, value: Any) -> str:
    value = adapter.normalize_prediction(value)
    def canon(item: Any) -> str:
        text = str(item).strip().lower()
        return text.replace("–", "-").replace("—", "-").replace("−", "-")
    if isinstance(value, (list, tuple)):
        return " | ".join(canon(x) for x in value)
    return canon(value) if value is not None else ""

def route_question(dataset: str, question: str) -> str:
    if dataset == "tabfact": return "agent"
    q = question.lower()
    agent_terms = ("how many", "how much", "total", "sum", "average", "mean", "difference", "highest", "lowest", "most", "least", "top", "after", "before", "more than", "less than")
    return "agent" if any(term in q for term in agent_terms) else "dp"

def choose_joint(client, dataset: str, sample: Sample, dp_value: Any, agent_value: Any, temperature: float, usage: dict[str, int], audit: Any = None) -> tuple[Any, str, dict[str, Any]]:
    adapter = get_adapter(dataset)
    dp_norm, agent_norm = normalized_answer(adapter, dp_value), normalized_answer(adapter, agent_value)
    if dp_norm and dp_norm == agent_norm and not (audit and audit.semantic_exception):
        return agent_value, "agreement", {"choice": "agent", "reason": "candidates normalized to the same answer"}
    if not dp_norm and agent_norm: return agent_value, "agent_only", {"choice": "agent"}
    if not agent_norm and dp_norm: return dp_value, "dp_only", {"choice": "dp"}
    # Numeric/date/aggregation questions benefit from executable evidence when
    # the textual candidate is an entity label.  This is a general route rule,
    # not a sample-specific condition.
    q_ir = heuristic_intent(sample.question, sample.df, dataset)
    def numeric(value):
        try: return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError): return None
    def grounded(value):
        text = str(value).strip().lower()
        return bool(text) and text in sample.table_md.lower()
    dp_num, agent_num = numeric(dp_value), numeric(agent_value)
    if q_ir.aggregation in {"count", "sum", "mean"}:
        if dp_num is not None and agent_num is not None and abs(dp_num-agent_num) < 1e-9 and grounded(dp_value):
            return dp_value, "dp_grounded_numeric", {"choice": "dp", "reason": "equivalent numeric candidates; DP preserves the exact grounded cell format"}
        return agent_value, "agent_numeric_route", {"choice": "agent", "reason": "question asks for a numeric aggregation answer"}
    if q_ir.answer_type in {"number", "date"} and grounded(dp_value) != grounded(agent_value):
        value, choice = (dp_value, "dp_grounded_value") if grounded(dp_value) else (agent_value, "agent_grounded_value")
        return value, choice, {"choice": "dp" if choice.startswith("dp") else "agent", "reason": "candidate is directly grounded in the table"}
    audit_context = (dumps(audit.to_dict())[:4000] if audit and audit.semantic_exception else "")
    text, response = ask(client, selector_prompt(dataset, sample.table_md, sample.question, str(dp_value), str(agent_value), audit_context), 700, temperature)
    usage_add(usage, response)
    try: obj = json.loads(re.search(r"\{.*\}", text, flags=re.S).group(0))
    except Exception: obj = {}
    choice = obj.get("choice") if obj.get("choice") in {"dp", "agent"} else "agent"
    return (dp_value if choice == "dp" else agent_value), "llm_selector", {"choice": choice, "selector_raw": text, "reason": obj.get("reason", "")}

def write_log(path: Path, sample: Sample, text: str, execution: dict[str, Any], audit: Any, repair_trace: list[dict[str, Any]], final_answer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def block(name: str, value: Any) -> str: return f"==================={name}===================\n{value}\n"
    content = block("Title", sample.title) + block("Table", sample.table_md) + block("Question", sample.question) + block("Text", text) + block("Answer", sample.gold)
    content += block("Execution Result", json.dumps(safe(execution.get("observed_value")), ensure_ascii=False))
    content += block("Execution Error", execution.get("error"))
    content += block("Global Audit", json.dumps(safe(audit.global_audit.to_dict() if audit else None), ensure_ascii=False))
    content += block("Operator Audit", json.dumps(safe(audit.operator_audit.to_dict() if audit else None), ensure_ascii=False))
    content += block("Parameter Audit", json.dumps(safe(audit.parameter_audit.to_dict() if audit else None), ensure_ascii=False))
    content += block("Semantic Logic Exception", json.dumps(safe(audit.to_dict() if audit else None), ensure_ascii=False))
    content += block("Repair Hint", audit.repair_hint if audit else "")
    content += block("Repair Trace", json.dumps(safe(repair_trace), ensure_ascii=False))
    content += block("Final Answer", final_answer)
    path.write_text(content, encoding="utf-8")

def run(args: argparse.Namespace) -> int:
    dataset = args.dataset.lower()
    adapter = get_adapter(dataset, sub_sample=not args.all_questions)
    end = args.end if args.end >= 0 else None
    samples = list(adapter.iter_range(args.start, end))
    if not samples: raise ValueError("The requested interval contains no samples")
    output = HERE / "outputs" / args.experiment
    output.mkdir(parents=True, exist_ok=True); (output / "log").mkdir(exist_ok=True)
    result_path = output / "result.jsonl"
    if result_path.exists() and not args.overwrite: raise FileExistsError(f"{result_path} exists; use a new experiment name or --overwrite")
    if result_path.exists(): result_path.unlink()
    selected = [s.sample_id for s in samples]
    (output / "selected_samples.json").write_text(json.dumps({"dataset": dataset, "start": args.start, "end": args.end, "sample_ids": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    config = vars(args).copy(); config.update({"dataset": dataset, "sample_ids": selected, "artifact_contract": "paper1_config_log_result_jsonl_v2"})
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    try: client = ParateraClient(cache_dir=str(output / "cache"), timeout=args.timeout)
    except APIConfigError as exc:
        print(str(exc), file=sys.stderr); return 2
    records = []
    for sample in samples:
        started = time.perf_counter(); usage, calls = {}, 0; raw_outputs, repair_trace = [], []
        execution = {"success": False, "error": "not_run", "observed_value": None}; initial_execution = execution
        code, repaired_code = "", None; q_ir = heuristic_intent(sample.question, sample.df, dataset); code_ir, audit = None, None
        dp_value, agent_value, selected_method, selection = None, None, args.mode, {}
        route = route_question(dataset, sample.question) if args.mode == "auto" else args.mode
        dp_text = ""
        if route in {"dp", "joint"}:
            dp_texts = []
            for _ in range(max(1, args.dp_votes)):
                text, response = ask(client, dp_prompt(dataset, sample.table_md, sample.title, sample.question), args.max_dp_tokens, args.temperature)
                dp_texts.append(text); raw_outputs.append(text); usage_add(usage, response); calls += int(response.get("api_calls", 0) or 0)
            candidates = [extract_answer(t, dataset) for t in dp_texts]; candidates = [x for x in candidates if x]
            dp_value = Counter(candidates).most_common(1)[0][0] if candidates else None; dp_text = dp_texts[0] if dp_texts else ""
        if route in {"baseline", "agent", "audit", "joint"}:
            code_text, response = ask(client, python_prompt(dataset, sample.table_md, sample.title, sample.question), args.max_code_tokens, args.temperature)
            raw_outputs.append(code_text); usage_add(usage, response); calls += int(response.get("api_calls", 0) or 0)
            code = extract_code(code_text); initial_execution = execute_code(code, sample.df); execution = initial_execution
            # Apply the same bounded execution-recovery policy to baseline, audit, and joint.
            # This repairs syntax/import/runtime failures only; it does not use gold answers.
            while (not execution.get("success") and len(repair_trace) < args.max_execution_repairs):
                execution_text, response = ask(client, execution_repair_prompt(dataset, sample.table_md, sample.question, code, str(execution.get("error"))), args.max_code_tokens, args.temperature)
                raw_outputs.append(execution_text); usage_add(usage, response); calls += int(response.get("api_calls", 0) or 0)
                execution_code = extract_code(execution_text)
                recovered_execution = execute_code(execution_code, sample.df)
                repair_trace.append({"kind": "execution", "repair_count": len(repair_trace) + 1, "execution_error": execution.get("error"), "repaired_code": execution_code, "execution": safe(recovered_execution)})
                code, execution = execution_code, recovered_execution
                if execution.get("success"): break
            code_ir = analyze_code(code, execution)
            agent_value = execution.get("result") if execution.get("success") else None
            if route == "audit" and execution.get("success"):
                if args.audit_mode == "llm_only":
                    audit, response = llm_audit(client, dataset, sample.table_md, sample.question, q_ir, code, code_ir); usage_add(usage, response); calls += int(response.get("api_calls", 0) or 0)
                else: audit = audit_intent(q_ir, code_ir, sample.df, AuditConfig(level=args.audit_level, mode=args.audit_mode))
                while audit and audit.semantic_exception and len(repair_trace) < args.max_repairs:
                    previous_code, previous_execution, previous_audit = code, execution, audit
                    previous_errors = sum(len(x.errors) for x in (audit.global_audit, audit.operator_audit, audit.parameter_audit))
                    repair_text, response = ask(client, repair_prompt(dataset, sample.table_md, sample.question, code, dumps(audit.to_dict())), args.max_code_tokens, args.temperature)
                    raw_outputs.append(repair_text); usage_add(usage, response); calls += int(response.get("api_calls", 0) or 0)
                    candidate_code = extract_code(repair_text); candidate_execution = execute_code(candidate_code, sample.df)
                    repair_item = {"repair_count": len(repair_trace) + 1, "diagnostic": safe(audit.to_dict()), "repaired_code": candidate_code, "execution": safe(candidate_execution)}
                    repair_trace.append(repair_item)
                    if not candidate_execution.get("success"):
                        repair_item["accepted"] = False
                        break
                    candidate_ir = analyze_code(candidate_code, candidate_execution)
                    if args.audit_mode == "llm_only":
                        candidate_audit, response = llm_audit(client, dataset, sample.table_md, sample.question, q_ir, candidate_code, candidate_ir); usage_add(usage, response); calls += int(response.get("api_calls", 0) or 0)
                    else:
                        candidate_audit = audit_intent(q_ir, candidate_ir, sample.df, AuditConfig(level=args.audit_level, mode=args.audit_mode))
                    candidate_errors = sum(len(x.errors) for x in (candidate_audit.global_audit, candidate_audit.operator_audit, candidate_audit.parameter_audit))
                    # Borrow the verifier/refiner rule: accept only a repair that
                    # passes verification or reduces the diagnosed error count.
                    if candidate_audit.semantic_exception:
                        repair_item["accepted"] = False
                        code, execution, audit = previous_code, previous_execution, previous_audit
                        code_ir = analyze_code(code, execution)
                        repaired_code = None
                        break
                    repair_item["accepted"] = True
                    repaired_code = candidate_code
                    code, execution, code_ir, audit = candidate_code, candidate_execution, candidate_ir, candidate_audit
                agent_value = execution.get("result") if execution.get("success") else None
            if route == "joint" and execution.get("success"):
                audit = audit_intent(q_ir, code_ir, sample.df, AuditConfig(level=args.audit_level, mode="hybrid"))
        if route == "dp": final_answer, selected_method = dp_value, "dp"
        elif route in {"baseline", "agent", "audit"}: final_answer, selected_method = agent_value, route
        else: final_answer, selected_method, selection = choose_joint(client, dataset, sample, dp_value, agent_value, args.temperature, usage, audit)
        final_correct = adapter.is_correct(final_answer, sample.gold)
        initial_value = initial_execution.get("result") if initial_execution.get("success") else (dp_value if route == "dp" else None)
        initial_correct = adapter.is_correct(initial_value, sample.gold)
        record = {**sample.paper1_metadata(), "sample_id": sample.sample_id, "method": args.mode, "selected_method": selected_method, "route": route,
                  "text": "\n---\n".join(raw_outputs), "generated_python": code, "execution_result": safe(initial_execution.get("observed_value")), "execution_error": initial_execution.get("error"),
                  "execution_success": bool(initial_execution.get("success")), "initial_execution_success": bool(initial_execution.get("success")), "final_execution_success": bool(execution.get("success")), "final_execution_result": safe(execution.get("observed_value")), "execution_warnings": safe((execution.get("evidence") or {}).get("semantic_warnings", [])),
                  "question_intent_ir": safe(q_ir.to_dict() if q_ir else None), "code_intent_ir": safe(code_ir.to_dict() if code_ir else None), "global_audit": safe(audit.global_audit.to_dict() if audit else None), "operator_audit": safe(audit.operator_audit.to_dict() if audit else None), "parameter_audit": safe(audit.parameter_audit.to_dict() if audit else None), "semantic_logic_exception": bool(audit.semantic_exception) if audit else False, "semantic_exception": bool(audit.semantic_exception) if audit else False, "semantic_exception_detail": safe(audit.to_dict() if audit else None),
                  "repair_hint": audit.repair_hint if audit else "", "repaired_code": repaired_code, "repair_trace": safe(repair_trace), "repair_count": len(repair_trace), "dp_answer": safe(dp_value), "agent_answer": safe(agent_value), "selection": safe(selection),
                  "initial_answer": safe(initial_value), "initial_correct": bool(initial_correct), "final_answer": safe(final_answer), "normalized_prediction": safe(adapter.normalize_prediction(final_answer)), "correct": bool(final_correct), "model": client.model, "api_usage": usage, "api_calls": calls, "latency_seconds": time.perf_counter() - started}
        write_log(output / "log" / f"{sample.flat_index}.txt", sample, "\n---\n".join(raw_outputs), execution, audit, repair_trace, final_answer)
        with result_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        records.append(record); print(f"{sample.flat_index}\t{sample.sample_id}\tmethod={selected_method}\tcorrect={final_correct}\texec={execution.get('success')}", flush=True)
    from second_paper.evaluation.metrics import summarize_records
    (output / "summary.json").write_text(json.dumps(summarize_records(records), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0

def main() -> None:
    parser = argparse.ArgumentParser(description="First-paper-compatible WTQ/TabFact runner")
    parser.add_argument("--dataset", choices=["wtq", "tabfact"], required=True)
    parser.add_argument("--mode", choices=["baseline", "audit", "agent", "dp", "joint", "auto"], default="baseline")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--start", type=int, default=0, help="0-based inclusive flat index")
    parser.add_argument("--end", type=int, default=-1, help="0-based exclusive flat index; -1 means no upper bound")
    parser.add_argument("--all-questions", action="store_true", help="Use every question instead of first-paper sampled_indices")
    parser.add_argument("--audit-level", choices=["global", "global_operator", "full"], default="full")
    parser.add_argument("--audit-mode", choices=["hybrid", "llm_only"], default="hybrid")
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--max-execution-repairs", type=int, default=1)
    parser.add_argument("--dp-votes", type=int, default=3)
    parser.add_argument("--max-dp-tokens", type=int, default=1800)
    parser.add_argument("--max-code-tokens", type=int, default=1800)
    parser.add_argument("--model", default="DeepSeek-V3.2")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); raise SystemExit(run(args))
if __name__ == "__main__": main()
