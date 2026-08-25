cot_prompt = """
You are an advanced AI capable of analyzing and understanding information within tables. Read the table below regarding "[TITLE]".

[TABLE]

Based on the given table, verify the following statement is true or false:

[QUESTION]

Please follow this format for your response:

Statement: [QUESTION]

Step 1: [First verification step - identify relevant data in the table]
Step 2: [Second verification step - extract and compare values]
Step 3: [Third verification step - logical analysis]
...
Step n: [Final verification step - conclude based on evidence]

Therefore, based on the table data, the statement is: [True/False]

Final Answer: Yes/No

Make sure to:
1. Break down your verification into clear, numbered steps
2. Reference specific table cells, rows, or columns in your reasoning
3. Show the logical connection between evidence and conclusion
4. The Final Answer must be exactly "Yes" or "No" only
"""