"""Dataset-specific adapters that preserve the first paper's data order and schema."""
from __future__ import annotations
from dataclasses import dataclass
import json, re, unicodedata
from pathlib import Path
from typing import Any, Iterator, Optional
import pandas as pd
from second_paper.paper1_reuse.wtq_utils import data as wtq_data
from second_paper.paper1_reuse.wtq_utils import execute as wtq_execute
from second_paper.paper1_reuse.wtq_utils import eval as wtq_eval
from second_paper.paper1_reuse.tabfact_utils import data as tabfact_data
from second_paper.paper1_reuse.tabfact_utils import execute as tabfact_execute
from second_paper.paper1_reuse.tabfact_utils import eval as tabfact_eval
PROJECT_ROOT = Path(__file__).resolve().parents[2]
@dataclass
class Sample:
    dataset: str; sample_id: str; table_id: str; question_id: str; question: str
    gold: Any; title: str; table_md: str; df: pd.DataFrame; flat_index: int
    def paper1_metadata(self) -> dict[str, Any]:
        return {"idx": self.flat_index, "answer": self.gold, "text": "", "transpose": False, "resort": [], "question_id": self.question_id, "table_id": self.table_id, "title": self.title, "table": self.table_md, "question": self.question}
class BaseAdapter:
    dataset = ""; data_dir_name = ""
    def __init__(self, sub_sample: bool = True) -> None:
        self.sub_sample = sub_sample
        path = PROJECT_ROOT / self.data_dir_name / "data" / f"{self.dataset}.json"
        with path.open(encoding="utf-8") as handle: self.raw = json.load(handle)
    def _table(self, record: dict[str, Any]) -> tuple[str, pd.DataFrame]: raise NotImplementedError
    def is_correct(self, prediction: Any, gold: Any) -> bool: raise NotImplementedError
    def normalize_prediction(self, prediction: Any) -> Any:
        return prediction.tolist() if isinstance(prediction, (pd.Series, pd.Index)) else prediction
    def iter_range(self, start: int = 0, end: Optional[int] = None) -> Iterator[Sample]:
        flat_index = 0
        for record in self.raw:
            all_indices = list(range(len(record.get("questions", []))))
            if self.sub_sample:
                sampled = set(record.get("sampled_indices") or [])
                indices = [idx for idx in all_indices if idx in sampled]
            else:
                indices = all_indices
            if not indices: continue
            table_md, df = self._table(record)
            table_id, title = str(record["table_id"]), str(record.get("title", ""))
            for idx in indices:
                if end is not None and flat_index >= end: return
                current = flat_index; flat_index += 1
                if current < start: continue
                question_id = str(record["ids"][idx])
                yield Sample(self.dataset, question_id, table_id, question_id, str(record["questions"][idx]), record["answers"][idx], title, table_md, df.copy(deep=True), current)
    def ids_in_range(self, start: int, end: Optional[int]) -> list[str]:
        return [sample.sample_id for sample in self.iter_range(start, end)]
def _canonical_wtq_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    # TableQA answers frequently contain en/em/minus dashes for ranges.
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text)


class WTQAdapter(BaseAdapter):
    dataset = "wtq"; data_dir_name = "Rethinking"
    def _table(self, record: dict[str, Any]) -> tuple[str, pd.DataFrame]:
        md = wtq_data.construct_markdown_table(**record["table"])
        return md, wtq_execute.convert_cells_to_numbers(wtq_execute.markdown_to_df(md))
    def is_correct(self, prediction: Any, gold: Any) -> bool:
        prediction = self.normalize_prediction(prediction)
        if prediction is None: return False
        if isinstance(prediction, (list, tuple)): prediction = ", ".join(str(x) for x in prediction)
        gold_text = "|".join(str(x) for x in gold) if isinstance(gold, list) else str(gold)
        return bool(wtq_eval.eval_ex_match(_canonical_wtq_text(prediction), _canonical_wtq_text(gold_text)))
class TabFactAdapter(BaseAdapter):
    dataset = "tabfact"; data_dir_name = "TabFact"
    def _table(self, record: dict[str, Any]) -> tuple[str, pd.DataFrame]:
        md = tabfact_data.construct_markdown_table(**record["table"])
        return md, tabfact_execute.convert_cells_to_numbers(tabfact_execute.markdown_to_df(md))
    def normalize_prediction(self, prediction: Any) -> Any:
        prediction = super().normalize_prediction(prediction)
        if isinstance(prediction, bool): return prediction
        if isinstance(prediction, (int, float)) and prediction in (0, 1): return bool(prediction)
        if prediction is None: return None
        text = str(prediction).strip().lower()
        if text in {"1", "yes", "true"}: return True
        if text in {"0", "no", "false"}: return False
        return tabfact_eval.normalize_tabfact_answer(text)
    def is_correct(self, prediction: Any, gold: Any) -> bool:
        if isinstance(gold, (int, bool)):
            target = bool(gold)
        else:
            text = str(gold).strip().lower()
            target = True if text in {"1", "yes", "true"} else False if text in {"0", "no", "false"} else tabfact_eval.normalize_tabfact_answer(text)
        return self.normalize_prediction(prediction) == target
def get_adapter(dataset: str, sub_sample: bool = True) -> BaseAdapter:
    normalized = dataset.lower()
    if normalized in {"wtq", "wikitablequestion"}: return WTQAdapter(sub_sample=sub_sample)
    if normalized in {"tabfact", "tabularfact"}: return TabFactAdapter(sub_sample=sub_sample)
    raise ValueError(f"Unsupported dataset: {dataset}")
