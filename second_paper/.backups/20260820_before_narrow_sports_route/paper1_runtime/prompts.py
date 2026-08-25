"""Dataset-specific prompts based on the first paper's prompt modules."""
from __future__ import annotations

import re


def _special_semantic_rules(dataset: str, table: str, question: str) -> str:
    """Return narrowly routed rules so unrelated questions keep the base prompt."""
    q = question.lower()
    header = table.splitlines()[0].lower() if table else ""
    rules: list[str] = []
    record_nouns = r"competitions?|films?|games?|players?|rows?|entries|records?"
    count_cue = r"how many|number of|total number|how many total"
    if re.search(count_cue, q) and re.search(rf"\b(?:{record_nouns})\b", q):
        rules.append(
            "Count matching dataframe rows. Repeated labels in different rows count separately. "
            "Do not use unique(), nunique(), drop_duplicates(), or set() unless the question "
            "explicitly says different, distinct, unique, or different kinds. Do not exclude "
            "rows because of a Notes/status value unless the question requests that exclusion."
        )
    if dataset == "tabfact":
        score_words = bool(re.search(r"\b(?:points?|goals?|score|scored)\b", q))
        score_header = bool(re.search(r"\b(?:score|agg|goals?)\b", header))
        points_header = bool(re.search(r"\bpoints?\b", header))
        if score_words and score_header and not points_header:
            rules.append(
                "In this sports score table, points, goals, score, and scored may describe the "
                "same displayed numeric score. Do not reject a numerically correct claim only "
                "because the scoring word differs."
            )
        if re.search(r"\b(?:object type|category|categories)\b", q) and re.search(
            r"\b(?:more|less|greater|fewer)\b", q
        ):
            rules.append(
                "Do not average repeated category rows unless the statement explicitly says "
                "average or mean. Test the exact row-level or pairwise relationship."
            )
    if not rules:
        return ""
    return "\nSPECIAL SEMANTIC RULES FOR THIS QUESTION:\n- " + "\n- ".join(rules)


def dp_prompt(dataset: str, table: str, title: str, question: str) -> str:
    if dataset == "tabfact":
        return f"""You are an advanced AI capable of analyzing information within tables. Read the table below regarding {title!r}.
{table}
Based on the given table, check the following statement as true or false:
{question}
Compare the statement with the exact row and column values. Keep numbers numeric,
respect the comparison operator and the statement's negation, and do not infer a
related but different statistic. Think step by step privately, then output exactly one final line in the form:
Final Answer: Yes
or
Final Answer: No
"""
    return f"""You are an advanced AI capable of analyzing information within tables. Read the table below regarding {title!r}.
{table}
Based on the given table, answer the following question:
{question}
Think step by step privately, then output the final answer in this exact form:
Final Answer: <short answer copied from the table>
"""


def baseline_python_prompt(dataset: str, table: str, title: str, question: str) -> str:
    answer_rule = "Return a real boolean True or False." if dataset == "tabfact" else "Return the requested answer."
    return f"""Use Python/pandas to answer the table question. A pandas DataFrame named df is already loaded.
Write one executable Python code block and assign the final result to answer. {answer_rule}
Do not call files or the network.
TITLE: {title}
TABLE:
{table}
QUESTION:
{question}
"""


def python_prompt(dataset: str, table: str, title: str, question: str) -> str:
    if dataset == "tabfact":
        rule, task = "The final answer must be a real boolean True or False.", "check whether the statement is true or false"
    else:
        rule, task = "The final answer may be a number, date, entity, or list of entities.", "answer the question"
    special_rules = _special_semantic_rules(dataset, table, question)
    return f"""You are working with a pandas dataframe named df. Use Python/pandas to {task}.
Write executable Python code only inside one ~~~python~~~ block. Do not import files or call the network.
The dataframe df is already loaded by the executor. NEVER recreate df, never write a data literal, and never copy the table into Python code. Use only df and ordinary Python/pandas operations.
Assign the final answer to a variable named answer. {rule}
Use exact column names and cell values from df. Apply every filter, comparator, aggregation, ranking, and arithmetic operation required by the question. If the question says different, distinct, or unique, count unique values rather than rows.
Return answer, not an intermediate Series, DataFrame, index, or diagnostic table. For WTQ, return one scalar for a singular question and a list only when the question explicitly asks for multiple answers. Preserve exact table text for entity/date/string answers. For TabFact, compare the claim directly against the relevant row(s), preserve whether the claim is negated, and return a real boolean rather than a yes/no string.
Do not hard-code a sample id, question text, or gold answer.{special_rules}
TITLE: {title}
TABLE SCHEMA/VALUES (reference only; do not reconstruct it):
{table}
QUESTION:
{question}
"""


def selector_prompt(dataset: str, table: str, question: str, dp_answer: str, agent_answer: str, audit_context: str = "") -> str:
    if dataset == "tabfact":
        instructions = "For TabFact, normalize Yes/No, True/False, and 1/0 to the same truth value before comparing. Decide only whether the exact statement is true in the table; never prefer a candidate merely because it is a Yes/No string. Re-check the exact row, number, comparator, and negation."
    else:
        instructions = "For WTQ, identify the answer requested by the question (entity, year/date, number, duration, or list) and do not substitute a related value from the same row. Respect ties and explicit plurality."
    return f"""Choose the more semantically correct answer for this table question.
Return only JSON: {{"choice":"dp" or "agent", "answer":"...", "reason":"..."}}.
Use the table as ground truth. {instructions} Do not invent values. If both candidates are plausible, compare their actual table-grounded semantics and requested cardinality.
AUDIT CONTEXT (deterministic evidence; warning only, never gold):
{audit_context}
TABLE:
{table}
QUESTION: {question}
DP CANDIDATE: {dp_answer}
PYTHON CANDIDATE: {agent_answer}
"""

def repair_prompt(dataset: str, table: str, question: str, code: str, diagnostic: str) -> str:
    return f"""Repair this Python/pandas program for the table question. Return only one executable ~~~python~~~ block and assign the final answer to answer.
The executor already provides df. NEVER recreate df, never copy table values into a data literal, and do not add unnecessary imports. Preserve correct existing operations; change only the operation/column/polarity identified by the diagnostic. Return answer, not a DataFrame or Series.
For TabFact, return a real boolean with the statement polarity preserved.
Do not hard-code a sample id, question, or gold answer. Apply the semantic diagnostic.
DATASET: {dataset}
TABLE SCHEMA/VALUES (reference only):
{table}
QUESTION:
{question}
CURRENT CODE:
~~
{code}
~~
SEMANTIC DIAGNOSTIC:
{diagnostic}
"""


def execution_repair_prompt(dataset: str, table: str, question: str, code: str, error: str) -> str:
    rule = "The final answer must be a real boolean True or False." if dataset == "tabfact" else "Return the requested scalar or list answer."
    return f"""Fix only the execution error in this pandas table-QA program. Return one executable ~~~python~~~ block and assign the final answer to answer.
The executor already provides df. NEVER recreate df or copy table values into code. Preserve the intended operations and answer semantics; do not invent values. {rule}
DATASET: {dataset}
TABLE SCHEMA/VALUES (reference only):
{table}
QUESTION: {question}
CURRENT CODE:
~~
{code}
~~
EXECUTION ERROR:
{error}
"""


def optimized_dp_prompt(dataset: str, table: str, title: str, question: str) -> str:
    if dataset == "tabfact":
        task = """Decide whether the exact statement is supported by the table.
Check every named entity, row, column, number, comparator, quantifier, and negation.
Do not replace the stated statistic with a related statistic. The answer must be Yes or No."""
    else:
        task = """Answer the exact table question.
First identify the requested answer type and cardinality. Then apply every entity filter,
 comparison, aggregation, ranking, arithmetic operation, and ordering constraint. Preserve
the exact table text for entity/date/duration answers and include all tied answers when asked."""
    special_rules = _special_semantic_rules(dataset, table, question)
    output_instruction = (
        f"Think through the relevant rows and operations privately.{special_rules}\n"
        "Output exactly one final line:"
        if special_rules
        else "Think through the relevant rows and operations privately. Output exactly one final line:"
    )
    return f"""You are the optimized direct-reasoning TableQA solver.
Use only the supplied table. Recompute the answer independently and do not use Python.
{task}
{output_instruction}
Final Answer: <answer>
TITLE: {title}
TABLE:
{table}
QUESTION:
{question}
"""


def dp_review_prompt(
    dataset: str,
    table: str,
    title: str,
    question: str,
    current_answer: str,
    previous_review: str = "",
) -> str:
    dataset_rule = (
        "The final answer is a truth value written as Yes or No. Re-check statement polarity and negation."
        if dataset == "tabfact"
        else
        "Preserve exact cell text for lookup answers; compute only when the question requires it. Respect singular versus plural and ties."
    )
    return f"""Act as an independent semantic auditor for a direct-reasoning TableQA answer.
Use only the table and question; no Python and no hidden reference answer.
Recompute the answer from scratch, then compare it with CURRENT ANSWER.
Check target column, entity filters, all comparators, aggregation/ranking/arithmetic,
answer type, cardinality, ordering, and exact value formatting. {dataset_rule}
Return one JSON object only:
{{"verdict":"pass" or "revise","answer":"final answer","confidence":0.0,
  "operations":["..."],"evidence":[{{"column":"...","value":"..."}}],
  "reason":"concise semantic reason"}}
Use revise only when the table provides concrete evidence that CURRENT ANSWER is wrong.
DATASET: {dataset}
TITLE: {title}
TABLE:
{table}
QUESTION: {question}
CURRENT ANSWER: {current_answer}
PREVIOUS REVIEW (may be empty):
{previous_review}
"""


def joint_reasoning_prompt(
    dataset: str,
    table: str,
    title: str,
    question: str,
    candidate_a: str,
    candidate_b: str,
    evidence_a: str,
    evidence_b: str,
) -> str:
    dataset_rule = (
        "For TabFact, decide the exact statement's truth value and return Yes or No; preserve negation and every comparator."
        if dataset == "tabfact"
        else
        "For WTQ, return the requested entity, date, duration, number, or complete list with table-exact formatting where applicable."
    )
    return f"""You are the final joint reasoner for two independently produced TableQA candidates.
The candidates and their diagnostics are fallible. Recompute the question from the table,
then compare both candidates against the exact target column, filters, operators, arithmetic,
ranking, cardinality, ties, and output type. {dataset_rule}
Do not favor a candidate because of its label or because it used Python. Use only this table.
If neither candidate is correct, return a recomputed answer.
Return one JSON object only:
{{"choice":"A" or "B" or "recompute","answer":"final answer","confidence":0.0,
  "evidence":[{{"column":"...","value":"..."}}],"reason":"concise semantic reason"}}
DATASET: {dataset}
TITLE: {title}
TABLE:
{table}
QUESTION: {question}
CANDIDATE A: {candidate_a}
CANDIDATE A EVIDENCE:
{evidence_a}
CANDIDATE B: {candidate_b}
CANDIDATE B EVIDENCE:
{evidence_b}

"""
