"""Dataset-specific prompts based on the first paper's prompt modules."""
from __future__ import annotations


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
    return f"""You are working with a pandas dataframe named df. Use Python/pandas to {task}.
Write executable Python code only inside one ~~~python~~~ block. Do not import files or call the network.
The dataframe df is already loaded by the executor. NEVER recreate df, never write a data literal, and never copy the table into Python code. Use only df and ordinary Python/pandas operations.
Assign the final answer to a variable named answer. {rule}
Use exact column names and cell values from df. Apply every filter, comparator, aggregation, ranking, and arithmetic operation required by the question. If the question says different, distinct, or unique, count unique values rather than rows.
Return answer, not an intermediate Series, DataFrame, index, or diagnostic table. For WTQ, return one scalar for a singular question and a list only when the question explicitly asks for multiple answers. Preserve exact table text for entity/date/string answers. For TabFact, compare the claim directly against the relevant row(s), preserve whether the claim is negated, and return a real boolean rather than a yes/no string.
Do not hard-code a sample id, question text, or gold answer.
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
