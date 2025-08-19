class DebugPrompts:
    #Please pay close attention to the descriptions of the columns in the DataFrame:
    #{data_annotation}
    basic_debug_prompt = """You are a helpful assistant that corrects the Python code that resulted in an error and returns the corrected code.

The code was designed to achieve this user request: '{input}'.
The DataFrame df that we are working with has already been defined and populated with the necessary data, so there is no need to load or create a new one.
The output of `print(df.head(2))` is:
{df}

The execution of the following code that was by a low-quality assistant resulted in an error:
```python
{code}
```

The error message was: "{error}".

Return only corrected Python code that fixes the error.
Use the same format with backticks.
If part of the code is not defined, or the whole code is a complete nonsense, define or rewrite it.
Here are examples of the output format:

Example format:
```python
import pandas as pd

def solve(df):
    # Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
    <CODE>

    # Extract the 5 rows from the sorted DataFrame.
    <CODE>

    # Create a list of the 'Country' column from the extracted DataFrame.
    <CODE>

    # Return the list of countries.
    return <RESULT>
```

i.e. in general, the format of your output must be
```python
<code>
```
The <code> must follow pep8 rules and must be syntactically correct! Do not use '`' character in <code> string. If the
error is due to "code_to_execute = Code.extract_code(regenerated_code, provider=self.provider)" it is a syntax error - make sure
to output syntactically correct code.

Your fixed code:
"""

    completion_debug_prompt = '''The code was designed to achieve this user request: '{input}'.
Here is the code that needs to be fixed:
## Faulty code:
```python
{code}
```

## Error message:
The error message is: "{error}"

Don't test the code by creating new DataFrames or print statements, only correct the code.

## Corrected solution:
{initial_coder_prompt}
'''