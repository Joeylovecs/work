"""Versioned prompts; baseline and audited method share the same code prompt."""
PROMPT_VERSION="v1_python_same_prompt"

def code_prompt(dataset: str, table: str, question: str) -> str:
    answer_rule="The answer must be a boolean (True/False)." if dataset=="tabfact" else "The answer may be a number, entity, date, or list."
    return f"""You are a pandas table question answering agent. The dataframe is named df.
Write executable Python code only inside one ```python``` block. Do not import files or call the network.
The code must assign the final answer to a variable named answer. {answer_rule}
Use the table and question exactly as provided. Avoid explanations outside the code block.
TABLE:
{table}
QUESTION:
{question}
"""

def intent_prompt(dataset: str, table: str, question: str) -> str:
    return f"""Extract a Question Intent IR for this {dataset} table question. Return JSON only with keys:
answer_type, cardinality, task_type, target_columns, filters, operations, aggregation, ranking, arithmetic, boolean_polarity.
Each filter must be {{\"column\": string, \"op\": one of \"=\", \"!=\", \">\", \">=\", \"<\", \"<=\", \"in\", \"not in\", \"value\": JSON scalar}}.
Do not invent columns or entities absent from the table.
TABLE:
{table}
QUESTION:
{question}
"""

def repair_prompt(dataset: str, table: str, question: str, code: str, diagnostic: str) -> str:
    return f"""Repair the Python program for a table question. Return only one executable ```python``` block and assign the final answer to answer.
Do not hard-code a sample id, question, or gold answer. Apply the general semantic diagnostic.
DATASET: {dataset}
TABLE:
{table}
QUESTION:
{question}
CURRENT CODE:
```python
{code}
```
SEMANTIC DIAGNOSTIC:
{diagnostic}
"""
