import pandas as pd
from second_paper.semantic_audit.ast_analyzer import analyze_code
from second_paper.semantic_audit.auditor import AuditConfig, audit_intent
from second_paper.semantic_audit.intent_schema import FilterSpec, QuestionIntent
from second_paper.semantic_audit.runtime_trace import execute_code
from second_paper.semantic_audit.question_intent import heuristic_intent
from second_paper.paper1_runtime.adapters import WTQAdapter, TabFactAdapter
from second_paper.paper1_runtime.prompts import python_prompt
from second_paper.scripts.run_guarded_joint import tabfact_semantic_route

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


def test_nunique_is_distinct_count_not_row_count():
    frame = pd.DataFrame({"Competition": ["World Cup", "World Cup", "Keirin"]})
    code = """answer = df["Competition"].nunique()"""
    execution = execute_code(code, frame)
    code_ir = analyze_code(code, execution, frame)
    assert code_ir.aggregation == "count_distinct"
    result = audit_intent(
        q(
            answer_type="number",
            task_type="aggregation",
            target_columns=["Competition"],
            operations=["aggregate"],
            aggregation="count",
        ),
        code_ir,
        frame,
    )
    assert has(result, "operator_audit", "WRONG_AGGREGATION")


def test_explicit_distinct_question_accepts_nunique():
    frame = pd.DataFrame({"Competition": ["World Cup", "World Cup", "Keirin"]})
    intent = heuristic_intent(
        "how many different competitions are listed?", frame, "wtq"
    )
    code = """answer = df["Competition"].nunique()"""
    execution = execute_code(code, frame)
    code_ir = analyze_code(code, execution, frame)
    result = audit_intent(intent, code_ir, frame)
    assert intent.aggregation == "count_distinct"
    assert not has(result, "operator_audit", "WRONG_AGGREGATION")


def test_total_number_of_numeric_column_is_sum_but_entity_column_is_count():
    frame = pd.DataFrame(
        {
            "Points": [40, 30, 25],
            "Competition": ["World Cup", "World Cup", "Keirin"],
        }
    )
    points = heuristic_intent("what was the total number of points?", frame, "wtq")
    competitions = heuristic_intent(
        "what is the total number of competition?", frame, "wtq"
    )
    assert points.aggregation == "sum"
    assert competitions.aggregation == "count"


def test_special_prompt_rules_are_routed_only_to_matching_semantics():
    table = "| Competition | Country |\n| World Cup | Denmark |"
    count_prompt = python_prompt(
        "wtq", table, "Cycling", "how many competitions were in Denmark?"
    )
    lookup_prompt = python_prompt(
        "wtq", table, "Cycling", "which country hosted the World Cup?"
    )
    assert "SPECIAL SEMANTIC RULES FOR THIS QUESTION" in count_prompt
    assert "Count matching dataframe rows" in count_prompt
    assert "SPECIAL SEMANTIC RULES FOR THIS QUESTION" not in lookup_prompt


def test_implicit_category_count_uses_distinct_values():
    frame = pd.DataFrame(
        {
            "League": ["ASL", "ASL", "ASL"],
            "Country": ["United Kingdom", "United Kingdom", "Denmark"],
        }
    )
    leagues = heuristic_intent(
        "what is the total number of soccer leagues that the team played in?",
        frame,
        "wtq",
    )
    countries = heuristic_intent(
        "what is the total number of countries on this list?", frame, "wtq"
    )
    assert leagues.aggregation == "count_distinct"
    assert countries.aggregation == "count_distinct"
    prompt = python_prompt(
        "wtq",
        "| League |\n| ASL |\n| ASL |",
        "Team",
        "what is the total number of soccer leagues that the team played in?",
    )
    assert "Count distinct category values" in prompt


def test_named_games_question_does_not_force_row_count_rule():
    prompt = python_prompt(
        "wtq",
        "| Year | Competition | Event |\n| 1999 | All-Africa Games | Discus |",
        "Athlete",
        "what is the number of all-africa games hannes has competed in?",
    )
    assert "Count matching dataframe rows" not in prompt


def test_participation_competition_count_uses_distinct_names():
    frame = pd.DataFrame(
        {"Competition": ["World Cup", "World Cup", "Keirin"]}
    )
    question = "what is the number of competitions hopley competed in?"
    intent = heuristic_intent(question, frame, "wtq")
    prompt = python_prompt(
        "wtq",
        "| Year | Competition | Event |\n"
        "| 2003 | All-Africa Games | Shot put |\n"
        "| 2003 | All-Africa Games | Discus |\n"
        "| 2007 | All-Africa Games | Discus |",
        "Athlete",
        question,
    )
    assert intent.aggregation == "count_distinct"
    assert "Count distinct competition editions" in prompt
    assert "time column together with Competition" in prompt
    assert "Count matching dataframe rows" not in prompt


def test_tabfact_joint_semantic_routes_are_narrow():
    category_table = "| object type | apparent magnitude |\n| irregular galaxy | 14.0 |"
    assert tabfact_semantic_route(
        "irregular galaxy is the object type having 2.1 more apparent magnitude than spiral galaxy",
        category_table,
    ) == "category_pair_comparison"
    assert tabfact_semantic_route(
        "all irregular galaxies have more apparent magnitude than spiral galaxies",
        category_table,
    ) is None
    assert tabfact_semantic_route(
        "the young africans scored two points",
        "| team 1 | agg |\n| young africans | 2 - 0 |",
    ) == "sports_score_alias"
    assert tabfact_semantic_route(
        "the winner had a 3-stroke margin",
        "| tournament | margin of victory |",
    ) is None


def test_shape_zero_count_and_negative_filter_are_audit_aligned():
    frame = pd.DataFrame(
        {
            "Competition": ["World Cup", "World Cup", "National Cup"],
            "Country": ["Denmark", "Denmark", "United Kingdom"],
        }
    )
    intent = heuristic_intent(
        "how many competitions were not in the united kingdom?", frame, "wtq"
    )
    code = """outside = df[df['Country'] != 'United Kingdom']
answer = outside.shape[0]"""
    execution = execute_code(code, frame)
    code_ir = analyze_code(code, execution, frame)
    result = audit_intent(intent, code_ir, frame)
    assert intent.filters[0].op == "!="
    assert code_ir.aggregation == "count"
    assert result.passed


def test_drop_duplicates_shape_zero_is_distinct_count():
    frame = pd.DataFrame(
        {
            "Year": [2003, 2003, 2007],
            "Competition": ["All-Africa Games"] * 3,
        }
    )
    code = """editions = df[['Year', 'Competition']].drop_duplicates()
answer = editions.shape[0]"""
    execution = execute_code(code, frame)
    code_ir = analyze_code(code, execution, frame)
    assert code_ir.aggregation == "count_distinct"


def test_incidental_len_does_not_hide_wrong_lookup_target():
    frame = pd.DataFrame(
        {
            "Year": [2007, 2008],
            "Competition": ["Games", "Championships"],
            "Venue": ["Algiers", "Addis Ababa"],
        }
    )
    intent = q(
        answer_type="string",
        task_type="lookup",
        target_columns=["Venue"],
        operations=[],
    )
    code = """latest = df[df['Year'] == df['Year'].max()]['Competition'].unique()
answer = latest[0] if len(latest) > 0 else None"""
    execution = execute_code(code, frame)
    code_ir = analyze_code(code, execution, frame)
    result = audit_intent(intent, code_ir, frame)
    assert has(result, "parameter_audit", "WRONG_TARGET_COLUMN")
