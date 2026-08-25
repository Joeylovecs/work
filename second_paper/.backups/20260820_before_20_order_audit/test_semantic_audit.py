import pandas as pd
from second_paper.semantic_audit.ast_analyzer import analyze_code
from second_paper.semantic_audit.auditor import AuditConfig, audit_intent
from second_paper.semantic_audit.intent_schema import FilterSpec, QuestionIntent
from second_paper.semantic_audit.runtime_trace import execute_code

DF=pd.DataFrame({"region":["East","West","East"],"year":[2021,2023,2024],"sales":[10,20,30],"profit":[1,2,3],"score":[5,9,2],"entity":["A","B","C"],"flag":[True,False,True]})
def audit(q,code):
    ex=execute_code(code,DF); ci=analyze_code(code,ex); return audit_intent(q,ci,DF,AuditConfig(level="full"))
def has(result,level,error): return any(e.error_type==error for e in getattr(result,level).errors)
def q(**kwargs): return QuestionIntent(**kwargs)

def test_sum_vs_mean_operator():
    r=audit(q(answer_type="number",task_type="aggregation",target_columns=["sales"],operations=["aggregate"],aggregation="sum"),'answer=df["sales"].mean()')
    assert has(r,"operator_audit","WRONG_AGGREGATION")
def test_strict_after_comparator_parameter():
    r=audit(q(answer_type="number",target_columns=["sales"],filters=[FilterSpec("year",">",2022)],operations=["filter","aggregate"],aggregation="sum"),'answer=df[df["year"]>=2022]["sales"].sum()')
    assert has(r,"parameter_audit","COMPARATOR_MISMATCH")
def test_wrong_target_column():
    r=audit(q(answer_type="number",target_columns=["sales"],operations=["aggregate"],aggregation="sum"),'answer=df["profit"].sum()')
    assert has(r,"parameter_audit","WRONG_TARGET_COLUMN")
def test_missing_filter_operator():
    r=audit(q(answer_type="number",target_columns=["sales"],filters=[FilterSpec("region","=","East"),FilterSpec("year",">",2022)],operations=["filter","filter","aggregate"],aggregation="sum"),'answer=df[df["region"]=="East"]["sales"].sum()')
    assert has(r,"operator_audit","MISSING_FILTER")
def test_wrong_entity_parameter():
    r=audit(q(answer_type="number",filters=[FilterSpec("region","=","East")],operations=["filter"],target_columns=["sales"]),'answer=df[df["region"]=="West"]["sales"].sum()')
    assert has(r,"parameter_audit","WRONG_ENTITY")
def test_wrong_sort_direction():
    r=audit(q(answer_type="entity",task_type="ranking",target_columns=["score"],operations=["ranking"],ranking={"direction":"asc","column":"score"}),'answer=df["score"].idxmax()')
    assert has(r,"operator_audit","WRONG_RANKING_DIRECTION")
def test_highest_implemented_as_sum():
    r=audit(q(answer_type="entity",task_type="ranking",target_columns=["score"],operations=["ranking"],ranking={"direction":"desc","column":"score"}),'answer=df["score"].sum()')
    assert has(r,"operator_audit","MISSING_RANKING") or has(r,"global_audit","FINAL_RETURN_SEMANTICS_MISMATCH")
def test_lowest_implemented_as_max():
    r=audit(q(answer_type="entity",task_type="ranking",target_columns=["score"],operations=["ranking"],ranking={"direction":"asc","column":"score"}),'answer=df["score"].max()')
    assert has(r,"operator_audit","MISSING_RANKING")
def test_answer_type_mismatch_global():
    r=audit(q(answer_type="entity",task_type="lookup",operations=[]),'answer=df["score"].sum()')
    assert has(r,"global_audit","ANSWER_TYPE_MISMATCH")
def test_cardinality_mismatch_global():
    r=audit(q(answer_type="entity",cardinality="multiple",operations=[]),'answer=df["entity"].iloc[0]')
    assert has(r,"global_audit","CARDINALITY_MISMATCH")
def test_arithmetic_sign_operator():
    r=audit(q(answer_type="number",task_type="comparison",operations=["arithmetic"],arithmetic="difference"),'answer=df["sales"].iloc[0]+df["sales"].iloc[1]')
    assert has(r,"operator_audit","ARITHMETIC_ERROR")
def test_boolean_polarity_parameter():
    r=audit(q(answer_type="boolean",task_type="boolean_verification",operations=["boolean_reduce"],boolean_polarity="true"),'answer=not (df["flag"].iloc[0] == True)')
    assert has(r,"parameter_audit","BOOLEAN_POLARITY_ERROR")
