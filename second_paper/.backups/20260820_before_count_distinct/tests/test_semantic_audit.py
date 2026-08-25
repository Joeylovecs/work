import pandas as pd
from second_paper.semantic_audit.ast_analyzer import analyze_code
from second_paper.semantic_audit.auditor import AuditConfig, audit_intent
from second_paper.semantic_audit.intent_schema import FilterSpec, QuestionIntent
from second_paper.semantic_audit.runtime_trace import execute_code
from second_paper.semantic_audit.question_intent import heuristic_intent
from second_paper.paper1_runtime.adapters import WTQAdapter, TabFactAdapter

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


def test_source_order_for_all_questions_and_sampled_subset():
    wtq_all = list(WTQAdapter(sub_sample=False).iter_range(0, 2))
    tabfact_all = list(TabFactAdapter(sub_sample=False).iter_range(0, 2))
    wtq_sampled = list(WTQAdapter(sub_sample=True).iter_range(0, 2))
    tabfact_sampled = list(TabFactAdapter(sub_sample=True).iter_range(0, 2))
    assert [sample.sample_id for sample in wtq_all] == ["nu-0", "nu-165"]
    assert [sample.sample_id for sample in tabfact_all] == [
        "1-24560733-1.html.csv_q0", "1-24560733-1.html.csv_q1"
    ]
    assert [sample.sample_id for sample in wtq_sampled] == ["nu-0", "nu-2928"]
    assert [sample.sample_id for sample in tabfact_sampled] == [
        "1-24560733-1.html.csv_q0", "1-24560733-1.html.csv_q1"
    ]


def test_duration_lookup_intent_is_table_grounded():
    frame = pd.DataFrame({
        "Cyclist": ["Alejandro Valverde (ESP)", "Paolo Bettini (ITA)"],
        "Time": ["5h 29' 10\"", "+0"],
    })
    intent = heuristic_intent(
        "how long did it take for alejandro valverde to finish?", frame, "wtq"
    )
    assert intent.answer_type == "duration"
    assert intent.target_columns == ["Time"]
    assert any(f.column == "Cyclist" and f.op == "contains" for f in intent.filters)
    assert intent.operations == ["filter", "select"]


def test_runtime_and_grounding_evidence_are_populated():
    frame = pd.DataFrame({
        "Cyclist": ["Alejandro Valverde (ESP)", "Paolo Bettini (ITA)"],
        "Time": ["5h 29' 10\"", "+0"],
    })
    code = """matched = df[df["Cyclist"].str.contains("Alejandro Valverde", case=False)]
answer = matched["Time"].iloc[0]"""
    execution = execute_code(code, frame)
    code_ir = analyze_code(code, execution, frame)
    assert "select" in code_ir.operations
    assert any(item.get("operation") == "filter" for item in code_ir.runtime_evidence if isinstance(item, dict))
    assert any(item.get("kind") == "entity" and item.get("passed") for item in code_ir.grounding_evidence if isinstance(item, dict))


def test_loc_indexer_is_not_reported_as_a_column():
    frame = pd.DataFrame({"Cyclist": ["Alejandro Valverde (ESP)"], "Time": ["5h"]})
    code = """answer = df.loc[df["Cyclist"].str.contains("Alejandro Valverde"), "Time"].iloc[0]"""
    execution = execute_code(code, frame)
    code_ir = analyze_code(code, execution, frame)
    assert "loc" not in code_ir.used_columns
    assert not any(item.get("requested") == "loc" for item in code_ir.grounding_evidence if isinstance(item, dict))


def test_empty_intermediate_dataframe_is_warned():
    frame = pd.DataFrame({"player": ["A"], "country": ["Australia"]})
    execution = execute_code(
        """filtered = df[(df["player"] == "a") & (df["country"] == "australia")]
answer = len(filtered) == 1""",
        frame,
    )
    assert "empty_intermediate_dataframe" in execution["evidence"]["semantic_warnings"]



def test_loc_target_column_drives_select_evidence():
    frame = pd.DataFrame({"Cyclist": ["Alejandro Valverde (ESP)"], "Time": ["5h"]})
    code = """answer = df.loc[df["Cyclist"].str.contains("Alejandro Valverde"), "Time"].iloc[0]"""
    execution = execute_code(code, frame)
    code_ir = analyze_code(code, execution, frame)
    assert "select:Time" in code_ir.ast_evidence
    assert not any(item.get("operation") == "select" and item.get("column") != "Time" for item in code_ir.runtime_evidence if isinstance(item, dict))



def test_sum_of_contains_boolean_mask_is_count():
    frame = pd.DataFrame({"Playoffs": ["Did not qualify", "Won", "Did not qualify"]})
    code = """answer = int(df["Playoffs"].str.contains("Did not qualify", na=False).sum())"""
    execution = execute_code(code, frame)
    code_ir = analyze_code(code, execution, frame)
    assert code_ir.aggregation == "count"
    result = audit_intent(
        q(
            answer_type="number",
            task_type="aggregation",
            target_columns=["Playoffs"],
            filters=[FilterSpec("Playoffs", "contains", "Did not qualify")],
            operations=["filter", "aggregate"],
            aggregation="count",
        ),
        code_ir,
        frame,
    )
    assert not has(result, "operator_audit", "WRONG_AGGREGATION")


def test_plural_question_noun_matches_singular_schema_column():
    frame = pd.DataFrame(
        {
            "Competition": ["World Cup", "World Cup"],
            "Country": ["United Kingdom", "Denmark"],
        }
    )
    intent = heuristic_intent(
        "how many competitions were not in the united kingdom?", frame, "wtq"
    )
    assert intent.target_columns == ["Competition"]
    assert intent.aggregation == "count"
    assert intent.source == "heuristic"
    assert intent.confidence >= 0.7


def test_total_number_of_entity_rows_is_count_not_sum():
    frame = pd.DataFrame({"Competition": ["World Cup", "World Cup", "Keirin"]})
    intent = heuristic_intent(
        "what is the total number of competition?", frame, "wtq"
    )
    assert intent.target_columns == ["Competition"]
    assert intent.aggregation == "count"


def test_unrequested_mean_is_rejected_for_row_level_comparison():
    frame = pd.DataFrame(
        {
            "object type": ["irregular galaxy", "irregular galaxy", "spiral galaxy"],
            "apparent magnitude": [14.0, 14.5, 11.9],
        }
    )
    code = """means = df.groupby("object type")["apparent magnitude"].mean()
answer = bool(means["irregular galaxy"] - means["spiral galaxy"] == 2.1)"""
    execution = execute_code(code, frame)
    code_ir = analyze_code(code, execution, frame)
    result = audit_intent(
        q(
            answer_type="boolean",
            task_type="comparison",
            target_columns=["apparent magnitude"],
            operations=["boolean_reduce"],
            aggregation=None,
        ),
        code_ir,
        frame,
    )
    assert has(result, "operator_audit", "UNREQUESTED_AGGREGATION")


def test_selected_column_is_recovered_through_answer_alias():
    frame = pd.DataFrame(
        {
            "Cyclist": ["Alejandro Valverde (ESP)", "Paolo Bettini (ITA)"],
            "Time": ["5h 29' 10\"", "+0"],
        }
    )
    code = """matched = df[df["Cyclist"].str.contains("Alejandro Valverde")]
time_value = matched["Time"].iloc[0]
answer = time_value"""
    execution = execute_code(code, frame)
    code_ir = analyze_code(code, execution, frame)
    assert "select" in code_ir.operations
    assert "select:Time" in code_ir.ast_evidence
    assert any(
        item.get("operation") == "select" and item.get("column") == "Time"
        for item in code_ir.runtime_evidence
        if isinstance(item, dict)
    )
