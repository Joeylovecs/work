"""Conservative gold-blind fusion that protects a valid preferred candidate."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paper1_runtime.adapters import get_adapter
from evaluation.metrics import summarize_records


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def invalid_answer(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text=value.strip()
        if not text:
            return True
        reduced=re.sub(r"(?i)^final\s+answer\s*:\s*", "", text).strip()
        return not reduced or reduced.lower() in {"none","null","nan","n/a","error","unknown"}
    return False


def norm(adapter, value: Any) -> str:
    value=adapter.normalize_prediction(value)
    if isinstance(value,(list,tuple)):
        return " | ".join(str(x).strip().lower() for x in value)
    return "" if value is None else str(value).strip().lower()


def run(args: argparse.Namespace) -> int:
    preferred=read_rows(Path(args.preferred))
    fallback=read_rows(Path(args.fallback))
    if len(preferred)!=len(fallback):
        raise ValueError("source row counts differ")
    if [r["question_id"] for r in preferred] != [r["question_id"] for r in fallback]:
        raise ValueError("sources are not aligned")
    adapter=get_adapter(args.dataset,sub_sample=False)
    out=Path(__file__).resolve().parents[1]/"outputs"/args.experiment
    out.mkdir(parents=True,exist_ok=True)
    result_path=out/"result.jsonl"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(result_path)
    if result_path.exists():
        result_path.unlink()
    config={
        **vars(args),
        "preferred":str(Path(args.preferred).resolve()),
        "fallback":str(Path(args.fallback).resolve()),
        "gold_excluded_from_selection":True,
        "policy":"keep any valid preferred answer; use fallback only for empty or malformed preferred answer",
    }
    (out/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    records=[]
    for a,b in zip(preferred,fallback):
        va,vb=a.get("final_answer"),b.get("final_answer")
        if norm(adapter,va) and norm(adapter,va)==norm(adapter,vb):
            final,choice,reason=va,"preferred","agreement"
        elif invalid_answer(va) and not invalid_answer(vb):
            final,choice,reason=vb,"fallback","preferred_invalid"
        else:
            final,choice,reason=va,"preferred","conservative_guard"
        record={
            "idx":a["idx"],
            "answer":a["answer"],
            "text":"",
            "transpose":a.get("transpose",False),
            "resort":a.get("resort",[]),
            "question_id":a["question_id"],
            "table_id":a.get("table_id"),
            "title":a.get("title",""),
            "table":a["table"],
            "question":a["question"],
            "sample_id":a.get("sample_id",a["question_id"]),
            "method":"conservative_fusion",
            "preferred_label":args.preferred_label,
            "fallback_label":args.fallback_label,
            "preferred_answer":va,
            "fallback_answer":vb,
            "preferred_correct":bool(adapter.is_correct(va,a["answer"])),
            "fallback_correct":bool(adapter.is_correct(vb,a["answer"])),
            "selected_method":choice,
            "selection":{"choice":choice,"reason":reason},
            "final_answer":final,
            "normalized_prediction":adapter.normalize_prediction(final),
            "correct":bool(adapter.is_correct(final,a["answer"])),
            "api_calls":0,
            "gold_excluded_from_selection":True,
        }
        with result_path.open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(record,ensure_ascii=False,default=str)+"\n")
        records.append(record)
        print(a["idx"],a["question_id"],choice,reason,record["correct"],flush=True)
    (out/"summary.json").write_text(json.dumps(summarize_records(records),ensure_ascii=False,indent=2),encoding="utf-8")
    return 0


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["wtq","tabfact"],required=True)
    parser.add_argument("--preferred",required=True)
    parser.add_argument("--fallback",required=True)
    parser.add_argument("--preferred-label",required=True)
    parser.add_argument("--fallback-label",required=True)
    parser.add_argument("--experiment",required=True)
    parser.add_argument("--overwrite",action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__=="__main__":
    main()
