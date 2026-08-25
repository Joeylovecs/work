"""Adapters that preserve paper-1 instance order, schema, serialization and evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import pandas as pd
from second_paper.paper1_reuse.wtq_utils import data as wtq_data, execute as wtq_execute, eval as wtq_eval
from second_paper.paper1_reuse.tabfact_utils import data as tf_data, execute as tf_execute, eval as tf_eval

ROOT=Path(__file__).resolve().parents[2]
@dataclass
class Sample:
    dataset: str; sample_id: str; table_id: str; question_id: str; question: str; gold: Any; title: str; table_md: str; df: pd.DataFrame
    def metadata(self):
        return {"dataset":self.dataset,"sample_id":self.sample_id,"table_id":self.table_id,"question_id":self.question_id,"question":self.question,"gold":self.gold,"title":self.title,"table":self.table_md}

def _root(dataset): return ROOT/('Rethinking' if dataset=="wtq" else 'TabFact')
def load_raw(dataset: str) -> List[Dict[str,Any]]:
    path=_root(dataset)/"data"/(dataset+".json")
    with path.open(encoding="utf-8") as f: return json.load(f)

def iter_samples(dataset: str, limit: Optional[int]=None, selected_ids: Optional[Iterable[str]]=None):
    dataset=dataset.lower(); raw=load_raw(dataset); wanted=set(selected_ids or []) if selected_ids else None; count=0
    for table in raw:
        indices=table.get("sampled_indices",list(range(len(table.get("questions",[])))))
        for idx in indices:
            sid=str(table["ids"][idx])
            if wanted is not None and sid not in wanted: continue
            if dataset=="wtq":
                table_obj=table["table"]; md=wtq_data.construct_markdown_table(**table_obj); df=wtq_execute.markdown_to_df(md); df=wtq_execute.convert_cells_to_numbers(df)
            else:
                table_obj=table["table"]; md=tf_data.construct_markdown_table(**table_obj); df=tf_execute.markdown_to_df(md); df=tf_execute.convert_cells_to_numbers(df)
            sample=Sample(dataset,sid,str(table["table_id"]),sid,str(table["questions"][idx]),table["answers"][idx],str(table.get("title","")),md,df)
            yield sample; count+=1
            if limit is not None and count>=limit: return

def sample_ids(dataset: str, limit: int) -> List[str]: return [s.sample_id for s in iter_samples(dataset,limit=limit)]

def _as_answer(value: Any) -> Any:
    if isinstance(value,dict) and "answer" in value: return value["answer"]
    return value

def normalize_prediction(dataset: str, value: Any) -> Any:
    value=_as_answer(value)
    if isinstance(value,(pd.Series,pd.Index)): value=value.tolist()
    if dataset=="tabfact":
        if isinstance(value,bool): return value
        if isinstance(value,(int,float)) and value in (0,1): return bool(value)
        return tf_eval.normalize_tabfact_answer(str(value)) if value is not None else value
    if isinstance(value,(list,tuple)): return [str(x) for x in value]
    return str(value) if value is not None else None

def is_correct(dataset: str, prediction: Any, gold: Any) -> bool:
    pred=normalize_prediction(dataset,prediction)
    if dataset=="tabfact":
        target=bool(gold) if isinstance(gold,(int,bool)) else tf_eval.normalize_tabfact_answer(str(gold))
        return pred==target
    if pred is None: return False
    pred_text=", ".join(pred) if isinstance(pred,list) else str(pred)
    gold_text="|".join(str(x) for x in gold) if isinstance(gold,list) else str(gold)
    return bool(wtq_eval.eval_ex_match(pred_text,gold_text))
