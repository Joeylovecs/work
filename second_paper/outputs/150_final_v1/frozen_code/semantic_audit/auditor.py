"""Three independent semantic alignment levels."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Optional
import pandas as pd
from .grounding import normalize_text, table_grounding, values_equal
from .intent_schema import AuditResult, CodeIntent, FilterSpec, LevelAudit, QuestionIntent, SemanticError

@dataclass
class AuditConfig:
    level: str = "full"  # global, global_operator, full
    mode: str = "hybrid"  # hybrid or llm_only; deterministic evidence is default
    enable_global: bool = True
    enable_operator: bool = True
    enable_parameter: bool = True
    def enabled(self,name: str) -> bool:
        if self.level=="global": return name=="global"
        if self.level=="global_operator": return name in {"global","operator"}
        if self.level in {"full","global_operator_parameter"}: return True
        return getattr(self,"enable_"+name,True)

def _err(level,typ,expected=None,actual=None,evidence=None,confidence=0.9,hint=""):
    return SemanticError(level=level,error_type=typ,expected=expected,actual=actual,evidence=list(evidence or []),confidence=confidence,repair_hint=hint)

def _has_op(ops: Iterable[str], expected: str) -> bool:
    return expected in set(ops or [])

def _filter_match(a: FilterSpec,b: FilterSpec, ignore_op=False) -> bool:
    op_equal=a.op==b.op or ({a.op,b.op}=={"=","contains"})
    return normalize_text(a.column)==normalize_text(b.column) and values_equal(a.value,b.value) and (ignore_op or op_equal)

def _hint(errors):
    return " ".join(e.repair_hint for e in errors if e.repair_hint)[:1500]

def _canonical_polarity(value: Any) -> Optional[str]:
    value=str(value or "").strip().lower()
    if value in {"positive","affirmative","assertive","true"}: return "positive"
    if value in {"negative","negated","not","explicit_negation","false"}: return "negative"
    return None

def _find_df_column(df: pd.DataFrame, name: str):
    target=normalize_text(name)
    for col in df.columns:
        if normalize_text(str(col))==target: return col
    return None

def _filter_holds(value: Any, op: str, expected: Any) -> bool:
    if op=="=": return values_equal(value,expected)
    if op=="!=": return not values_equal(value,expected)
    try:
        left=float(value); right=float(expected)
        return {">":left>right, ">=":left>=right, "<":left<right, "<=":left<=right}.get(op,False)
    except (TypeError,ValueError):
        left=normalize_text(value); right=normalize_text(expected)
        return {">":left>right, ">=":left>=right, "<":left<right, "<=":left<=right}.get(op,False)

def _implied_by_code_filters(qf: FilterSpec, code_filters, df: Optional[pd.DataFrame]) -> bool:
    """Return true only when code filters select rows that all satisfy qf."""
    if df is None or df.empty or not code_filters: return False
    # A same-column filter with a different comparator is not redundancy; it is
    # precisely the comparator mismatch the parameter audit must expose.
    if any(normalize_text(cf.column)==normalize_text(qf.column) for cf in code_filters): return False
    mask=pd.Series(True,index=df.index)
    for cf in code_filters:
        col=_find_df_column(df,cf.column)
        if col is None: return False
        mask &= df[col].map(lambda value: _filter_holds(value,cf.op,cf.value))
    if not bool(mask.any()): return False
    col=_find_df_column(df,qf.column)
    if col is None: return False
    return bool(df.loc[mask,col].map(lambda value: _filter_holds(value,qf.op,qf.value)).all())


def audit_intent(question: QuestionIntent, code: CodeIntent, df: Optional[pd.DataFrame]=None, config: Optional[AuditConfig]=None) -> AuditResult:
    config=config or AuditConfig()
    ga=LevelAudit("GLOBAL",passed=True); oa=LevelAudit("OPERATOR",passed=True); pa=LevelAudit("PARAMETER",passed=True)
    if config.enabled("global"):
        # Table answers are frequently serialized as strings even when the
        # question asks for an entity, date, duration, or numeric value.  Treat
        # these scalar representations as compatible; reserve this check for
        # real cardinality/object mismatches.
        compatible_pairs={("entity", "entity"), ("entity", "string"), ("number", "number"), ("number", "string"), ("date", "date"), ("date", "string"), ("duration", "duration"), ("duration", "string"), ("scalar", "scalar"), ("scalar", "string")}
        scalar_compatible=(question.answer_type, code.return_type) in compatible_pairs
        if question.answer_type not in {"unknown",""} and code.return_type not in {"unknown",""} and question.answer_type!=code.return_type and not scalar_compatible:
            ga.errors.append(_err("GLOBAL","ANSWER_TYPE_MISMATCH",question.answer_type,code.return_type,code.ast_evidence,0.98,"Return the requested answer type, not an intermediate numeric or table object."))
        if question.answer_type != "boolean" and question.cardinality not in {"unknown",""} and code.cardinality not in {"unknown",""} and question.cardinality!=code.cardinality:
            ga.errors.append(_err("GLOBAL","CARDINALITY_MISMATCH",question.cardinality,code.cardinality,code.runtime_evidence,0.95,"Return one answer or the complete list required by the question."))
        if question.task_type=="ranking" and question.answer_type != "boolean" and not code.ranking:
            ga.errors.append(_err("GLOBAL","FINAL_RETURN_SEMANTICS_MISMATCH","ranked entity",code.return_type,code.ast_evidence,0.9,"Return the entity selected by the ranking operation."))
    if config.enabled("operator"):
        expected_ops=list(question.operations or [])
        actual_ops=list(code.operations or [])
        if len(question.filters)>len(code.filters):
            matched=[]
            for qf in question.filters:
                if any(_filter_match(qf,cf,ignore_op=True) for cf in code.filters): matched.append(qf)
            for qf in question.filters:
                if qf not in matched and not _implied_by_code_filters(qf,code.filters,df):
                    oa.errors.append(_err("OPERATOR","MISSING_FILTER",qf.to_dict(),[x.to_dict() for x in code.filters],code.ast_evidence,0.96,f"Add filter {qf.column} {qf.op} {qf.value} before the aggregation."))
        for expected in expected_ops:
            equivalent=(expected=="count" and code.aggregation=="count") or (expected in {"max","min"} and ("compare" in actual_ops or code.ranking is not None)) or (expected=="group_by" and "group_by" in actual_ops) or (expected=="boolean_reduce" and code.return_type=="boolean") or (expected=="ranking" and question.answer_type=="boolean" and ("compare" in actual_ops or code.aggregation is not None))
            if expected not in actual_ops and expected not in {"select"} and not equivalent:
                oa.errors.append(_err("OPERATOR","MISSING_OPERATOR",expected,actual_ops,code.ast_evidence,0.93,f"Add the missing {expected} operation before returning the answer."))
        if question.aggregation and code.aggregation and question.aggregation!=code.aggregation:
            oa.errors.append(_err("OPERATOR","WRONG_AGGREGATION",question.aggregation,code.aggregation,code.ast_evidence,0.98,f"Use {question.aggregation} rather than {code.aggregation}."))
        # Averaging is never a neutral way to collapse repeated category rows.
        # If the statement does not explicitly request an average/mean, a mean()
        # can turn an existential or row-level comparison into a different claim.
        if not question.aggregation and code.aggregation == "mean":
            oa.errors.append(_err(
                "OPERATOR",
                "UNREQUESTED_AGGREGATION",
                None,
                "mean",
                code.ast_evidence,
                0.98,
                "Do not average rows unless the question explicitly asks for an average or mean. Test the stated row-level or category relationship.",
            ))
        if question.aggregation and "aggregate" in expected_ops and not code.aggregation:
            oa.errors.append(_err("OPERATOR","MISSING_AGGREGATION",question.aggregation,None,code.ast_evidence,0.96,f"Apply {question.aggregation} to the requested target column."))
        if isinstance(question.ranking,dict) and question.ranking and not code.ranking:
            oa.errors.append(_err("OPERATOR","MISSING_RANKING",question.ranking,None,code.ast_evidence,0.96,"Use the requested ranking or argmax/argmin operation."))
        if isinstance(question.ranking,dict) and question.ranking and isinstance(code.ranking,dict):
            exp_dir=question.ranking.get("direction"); act_dir=code.ranking.get("direction")
            if exp_dir and act_dir and exp_dir!=act_dir:
                oa.errors.append(_err("OPERATOR","WRONG_RANKING_DIRECTION",exp_dir,act_dir,code.ast_evidence,0.96,"Reverse the ranking direction to match highest/lowest."))
        if question.arithmetic and code.arithmetic and question.arithmetic!=code.arithmetic:
            oa.errors.append(_err("OPERATOR","ARITHMETIC_ERROR",question.arithmetic,code.arithmetic,code.ast_evidence,0.94,"Use the requested arithmetic operation and sign."))
    if config.enabled("parameter"):
        grounded_columns=[c for c in code.used_columns if normalize_text(c) not in {normalize_text(x) for x in code.derived_columns}]
        grounding=table_grounding(df,grounded_columns,code.filters) if df is not None else {"unknown_columns":[],"aliases":{},"entities_missing":[],"evidence":[]}
        pa.errors.extend(_err("PARAMETER","UNKNOWN_COLUMN",None,col,grounding.get("evidence",[]),0.99,"Use an existing table column.") for col in grounding.get("unknown_columns",[]))
        for qf in question.filters:
            exact=next((cf for cf in code.filters if _filter_match(qf,cf)),None)
            if exact or _implied_by_code_filters(qf,code.filters,df): continue
            same_value=next((cf for cf in code.filters if normalize_text(cf.column)==normalize_text(qf.column) and values_equal(cf.value,qf.value)),None)
            same_col=next((cf for cf in code.filters if normalize_text(cf.column)==normalize_text(qf.column)),None)
            if same_value:
                pa.errors.append(_err("PARAMETER","COMPARATOR_MISMATCH",qf.op,same_value.op,[f"question_filter={qf.to_dict()}",f"code_filter={same_value.to_dict()}"],0.99,f"Use comparator {qf.op} for {qf.column} {qf.value}."))
            elif same_col:
                pa.errors.append(_err("PARAMETER","WRONG_ENTITY",qf.value,same_col.value,[f"question_filter={qf.to_dict()}",f"code_filter={same_col.to_dict()}"],0.96,"Use the exact entity or cell value requested by the question."))
            else:
                pa.errors.append(_err("PARAMETER","MISSING_FILTER_PARAMETER",qf.to_dict(),None,code.ast_evidence,0.9,"Ground the filter column and value in the question and table."))
        # For boolean verification, the target value may be consumed inside a
        # reduction/comparison and need not appear as the final selected column.
        check_targets=(
            not (
                question.answer_type == "boolean"
                and question.task_type
                in {"verification", "boolean_verification", "comparison"}
            )
            and not (code.aggregation == "count" and code.return_type == "number")
            and "dynamic_column_access" not in code.ast_evidence
        )
        if check_targets:
            for target in question.target_columns:
                if code.used_columns and not any(normalize_text(target)==normalize_text(c) for c in code.used_columns):
                    pa.errors.append(_err("PARAMETER","WRONG_TARGET_COLUMN",target,code.used_columns,code.ast_evidence,0.94,"Use the target column named in the question."))
        if question.answer_type=="boolean" and question.boolean_polarity and code.boolean_polarity:
            qpol=_canonical_polarity(question.boolean_polarity)
            cpol=_canonical_polarity(code.boolean_polarity)
            # A negative statement can be implemented by a reversed comparator
            # or an all/any reduction without a literal `not`. Only flag the
            # unambiguous inversion where positive intent is negated explicitly.
            if qpol=="positive" and cpol=="negative":
                pa.errors.append(_err("PARAMETER","BOOLEAN_POLARITY_ERROR",question.boolean_polarity,code.boolean_polarity,code.ast_evidence,0.9,"Preserve the truth polarity of the statement; do not invert the comparison."))
        if isinstance(question.ranking,dict) and isinstance(code.ranking,dict) and question.ranking.get("column") and code.ranking.get("column"):
            if normalize_text(question.ranking["column"])!=normalize_text(str(code.ranking["column"])):
                pa.errors.append(_err("PARAMETER","WRONG_SORT_COLUMN",question.ranking["column"],code.ranking["column"],code.ast_evidence,0.94,"Rank using the requested target column."))
    for level in (ga,oa,pa):
        level.passed=not level.errors
        level.confidence=min((e.confidence for e in level.errors),default=1.0)
    selected=[x for x,enabled in ((ga,config.enabled("global")),(oa,config.enabled("operator")),(pa,config.enabled("parameter"))) if enabled]
    errors=[e for x in selected for e in x.errors]
    return AuditResult(passed=not errors,semantic_exception=bool(errors),global_audit=ga,operator_audit=oa,parameter_audit=pa,repair_hint=_hint(errors),confidence=min((e.confidence for e in errors),default=1.0),mode=config.mode)
