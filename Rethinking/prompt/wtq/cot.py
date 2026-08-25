cot_prompt = """
You are an advanced AI capable of analyzing and understanding information within tables. Read the table below regarding "[TITLE]".

[TABLE]

Based on the given table, answer the following question:

[QUESTION]

Please follow this format for your response:

Question: [QUESTION]

Step 1: [First reasoning step]
Step 2: [Second reasoning step]
Step 3: [Third reasoning step]
...
Step n: [Final reasoning step]

Therefore, the final answer is: [Your conclusion]

Final Answer: AnswerName1, AnswerName2...

Make sure to break down your reasoning into clear, numbered steps and provide a logical conclusion before giving the final answer.
"""