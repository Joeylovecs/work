import pandas as pd

def construct_markdown_table(header: list, rows: list, header_aware: bool = False, **kwargs) -> str:
    """
    Construct a markdown table from a header and rows.

    Parameters:
    header (list): The header of the table.
    rows (list): The rows of the table.
    header_aware (bool): Whether the table is header aware. Defaults to False.

    Returns:
    str: The markdown table.
    """
    
    def escape_cell(cell):
        """转义单元格中的特殊字符"""
        if isinstance(cell, str):
            # 替换管道符为全角字符或用反斜杠转义
            return cell.replace('|', '\|')
        return str(cell)

    table = ""
    # if the header is a list of strings, then the header is first row of the table
    if isinstance(header[0], str):
        table += "| " + " | ".join([escape_cell(h) for h in header]) + " |\n"
        if header_aware:
            table += "| " + " | ".join(["---"] * len(header)) + " |\n"
        
        # iterate over the rows
        for row in rows:
            table += "| " + " | ".join([escape_cell(cell) for cell in row]) + " |\n"
    
    # if the header is a list of lists, then the header is the first column of the table
    elif isinstance(header[0], list):
        # add the header to each row
        rows = [header[i] + row for i, row in enumerate(rows)]

        # iterate over the rows
        for row in rows:
            table += "| " + " | ".join([escape_cell(cell) for cell in row]) + " |\n"
        
    return table.strip()

def print_partial_markdown(df, keep: int=3):
    """
    Print a partial markdown table.
    
    Parameters:
    df (pd.DataFrame): The dataframe to print.
    keep (int): The number of rows to keep. Defaults to 3.
    
    Returns:
    str: The partial markdown table.
    """
    # Concatenate the first `keep` and last `keep` rows of the dataframe
    combined_df = pd.concat([df.head(keep), df.tail(keep)])
    
    # Convert the combined dataframe to markdown
    markdown_output = combined_df.to_markdown(index=False)
    
    # Insert the "..." separator in the appropriate line
    markdown_lines = markdown_output.split('\n')
    separator_index = len(df.head(keep).to_markdown(index=False).split('\n'))
    markdown_lines.insert(separator_index, '...')
    
    # Join the lines to form the final markdown string and print
    final_output = '\n'.join(markdown_lines)
    
    return final_output