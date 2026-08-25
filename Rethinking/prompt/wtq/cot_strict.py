cot_prompt_strict = """
You are an advanced AI capable of analyzing and understanding information within tables. Read the table below regarding "[TITLE]".

[TABLE]

Based on the given table, answer the following question:

[QUESTION]

IMPORTANT FORMATTING REQUIREMENTS:
1. DO NOT use <think> tags or any internal thinking processes
2. MUST follow the exact step-by-step format below
3. Each step should be concise and focused (maximum 3 sentences per step)
4. MUST include "Therefore, the final answer is:" before conclusion
5. MUST end with "Final Answer:" followed by the exact answer

Required Format:

Question: [QUESTION]

Step 1: [First reasoning step - be concise, max 3 sentences]
Step 2: [Second reasoning step - be concise, max 3 sentences]
Step 3: [Third reasoning step - be concise, max 3 sentences]
...
Step n: [Final reasoning step - be concise, max 3 sentences]

Therefore, the final answer is: [Your conclusion]

Final Answer: [Exact answer]

Example of good format:
Question: What is the highest value?
Step 1: Examine the values in the data column to identify all numerical entries.
Step 2: Compare all numerical values systematically to find the maximum. Check each row to ensure no values are missed.
Step 3: Identify that 25.7 is the highest value, located in row 3 of the table.
Therefore, the final answer is: 25.7
Final Answer: 25.7

Follow this format EXACTLY. Do not deviate from this structure.
"""
