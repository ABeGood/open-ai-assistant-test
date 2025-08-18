from .base_prompt_strategy import BasePromptStrategy
import re
from .plotting_guidelines import seaborn_implementation

class HybridPromptsForFunctionAndTextGeneration(BasePromptStrategy):

    has_planner = False
    max_debug_times = 2

    generate_code_gpt4_ = """You are highly proficient with Python and the pandas library. A user has submitted a query that you need to address: '{{input}}'. 
You also have a list of subtasks that need to be completed.

A pandas DataFrame named `df` has been provided. It has already been defined and populated with the necessary data, so there is no need to load or recreate it.
The output of `print(df.head({{head_number}}))` is:
{{df_head}}

Please pay close attention to the descriptions of the columns in the DataFrame:
{{column_description}}

Adhere to these visualisation guidelines:
{seaborn_best_practices}

Your task is to define a Python function named `def solve(df):` that fulfills the user's query and returns the result of the analysis.
Include all necessary import statements at the beginning of the code.
Use backticks to enclose the code.
Only include visualizations if the user's query explicitly requests them (e.g., plot, chart).
Do not test the function with commands like `print(solve(df))`; only define the function. Avoid including code descriptions or usage examples.
Ensure not to check the instance type against `int` or `float`; use `Number` instead.
Be cautious of data order—avoid changing it through operations like sorting.[]
Do not use `reset_index()`. Leave the DataFrame index unaltered after processing.
Before returning the result in the `solve` function, remove all auxiliary or temporary columns (e.g., a 'delta' column).
Be cautious of NaN, Inf and other special values - the table contains few of them.
Avoid using pd.DataFrame.reset_index() method, it may cause failure in our unittests.

Here are examples of the output format:

Example 1 (top five selection - ordering does not matter):
```python
import pandas as pd

def solve(df):
    # Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
    sorted_df = df.sort_values(by='happiness_index', ascending=False)

    # Extract the top 5 rows from the sorted DataFrame.
    top_five = sorted_df.head(5)

    # Create a list of the 'Country' column from these rows.
    countries = top_five['Country'].tolist()

    # Return the list of countries.
    return countries
```

Example 2 (top five selection - ordering does matter):
```
import pandas as pd

def solve(df):
    # select top 5, keep the current order
    top_five = df.nlargest(5, 'happiness_index')
    
    # Create a list of the 'Country' column from these rows.
    countries = top_five['Country'].tolist()
    
    # Return the list of countries.
    return countries
```

Example 3:
```python
import pandas as pd

def solve(df):
    # Filter the DataFrame `df` for entries where the 'age' column is greater than 30.
    filtered_df = df[df['age'] > 30]

    # Calculate the average 'salary' for these entries.
    average_salary = filtered_df['salary'].mean()

    # Return the average salary.
    return average_salary
```

i.e. in general, the format of your output must be
```python
<code>
```
The <code> must follow pep8 rules and must be syntactically correct! Do not use '`' character in <code> string.

Your code:
""".format(seaborn_best_practices=seaborn_implementation)

    generate_code = """You are really good with Python and the pandas library. The user provided a query that you need to help achieve: '{{input}}'. 
You also have a list of subtasks to be accomplished.

You have been presented with a pandas DataFrame named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({{head_number}}))` is:
{{df_head}}
{{column_description}}

{{functions_description}}

Adhere to these visualisation guidelines:
{seaborn_best_practices}

Return the definition of a Python function called `def solve(df):` that accomplishes the user query and returns the result of the analysis.
You must include the necessary import statements at the top of the code.
You must use the backticks to enclose the code.
Only show the visualizations if the user query explicitly asks for them (plot, chart, etc.).
Do not test the function with anything similar to `print(solve(df))`, only define the function, in the format of the following example:

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
""".format(seaborn_best_practices=seaborn_implementation)

    generate_code_multiple_dfs = """You are really good with Python and the pandas library. The user provided a query that you need to help achieve: '{{input}}'. 
You also have a list of subtasks to be accomplished.

You have been presented with several pandas DataFrames named `df_k`, where k is a number; e.g. `{{df_examples}}` etc.
The DataFrames `df_k` have already been defined and populated with the required data, so don't load it and don't create new ones.
Here are the first 5 rows of the given DataFrames, sorted (i.e. first 5 rows of `df_1`, then `df_2`, etc.):
{{data}}

{{functions_description}}

Adhere to these visualisation guidelines:
{seaborn_best_practices}

Return the definition of a Python function called `def solve({{df_examples}}):` that accomplishes the user query and returns the result of the analysis.
You must include the necessary import statements at the top of the code.
You must use the backticks to enclose the code.
Only show the visualizations if the user query explicitly asks for them (plot, chart, etc.).
All dataframes are already loaded, and defined. Do not define any other dataframe or table! Unused dataframes should be passed as optional arguments to the solve function.
Do not test the function with anything similar to `print(solve({{df_examples}}))`, only define the function, in the format of the following example:

Here are examples of the output format:

Example format:
```python
import pandas as pd

def solve({{df_examples}}):
    # Step 1: some comment.
    <CODE>

    # STEP 2: some comment.
    <CODE>

    # Step 3: some comment.
    <CODE>

    # Return the result.
    return <RESULT>
```
""".format(seaborn_best_practices=seaborn_implementation)

    generate_code_multiple_dfs_plot = """You are really good with Python and the pandas library. The user provided a query that you need to help achieve: '{{input}}'. 
You also have a list of subtasks to be accomplished.

You have been presented with several pandas DataFrames named `df_k`, where k is a number; e.g. `{{df_examples}}` etc.
The DataFrames `df_k` have already been defined and populated with the required data, so don't load it and don't create new ones.
Here are the first 5 rows of the given DataFrames, sorted (i.e. first 5 rows of `df_1`, then `df_2`, etc.):
{{data}}

{{functions_description}}

Adhere to these visualisation guidelines:
{seaborn_best_practices}


Return the definition of a Python function called `def solve({{df_examples}}):` that accomplishes the user query and returns the result of the analysis.
You must include the necessary import statements at the top of the code.
You must use the backticks to enclose the code.
Only show the visualizations if the user query explicitly asks for them (plot, chart, etc.).
All dataframes are already loaded, and defined. Do not define any other dataframe or table! Unused dataframes should be passed as optional arguments to the solve function.
Do not test the function with anything similar to `print(solve({{df_examples}}))`, only define the function, in the format of the following example:
You must not include `plt.show()`. Just save the plot to '{{save_plot_name}}' with `plt.savefig('{{save_plot_name}}')`.

Here are examples of the output format:

Example format:
```python
import pandas as pd
import matplotlib.pyplot as plt

def solve({{df_examples}}):
    # Step 1: some comment.
    <CODE>

    # STEP 2: some comment.
    <CODE>

    # Step 3: some comment.
    <CODE>

    # Save the pie plot to 'plots/example_plot00.png'.
    save and return
```
Your code:
""".format(seaborn_best_practices=seaborn_implementation)

    generate_code_for_plot_save = """You are really good with Python and the pandas library. The user provided a query that you need to help achieve: '{{input}}'. 

You have been presented with a pandas DataFrame named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({{head_number}}))` is:
{{df_head}}
{{column_description}}

{{functions_description}}

Adhere to these visualisation guidelines:
{seaborn_best_practices}


Return the definition of a Python function called `def solve(df):` that accomplishes the user query and returns the result of the analysis.
You must include the necessary import statements at the top of the code.
You must not include `plt.show()`. Just save the plot to '{{save_plot_name}}' with `plt.savefig('{{save_plot_name}}')`.
You must use the backticks to enclose the code.
Do not test the function with anything similar to `print(solve(df))`, only define the function, in the format of the following example.

Here is an example of the output format:
```python
import pandas as pd
import matplotlib.pyplot as plt

def solve(df):
    # Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
    <CODE>

    # Extract the 5 rows from the sorted DataFrame.
    <CODE>

    # Create a pie plot of the 'GDP' column of the extracted DataFrame.
    <CODE>

    # Save the pie plot to 'plots/example_plot00.png'.
    <CODE>
```
""".format(seaborn_best_practices=seaborn_implementation)

    generate_whatever = """
You are highly proficient with data Python analyst and you are primarily using the pandas library. A user has submitted a query that you need to address: '{{input}}'. 
I.e. your task is to answer the query, either by providing the code, textual answer or both. If the user asks a 'how to' question, 
you must provide textual explanation to the methods you are using. If the user asks for summary, you must provide the summary (i.e. text plus code, not just code snipets).

A pandas DataFrame named `df` has been provided. It has already been defined and populated with the necessary data, so there is no need to load or recreate it.
The output of `print(df.head({{head_number}}))` is:
{{df_head}}

Please pay close attention to the descriptions of the columns in the DataFrame:
{{column_description}}

Follow these guidelines:
- your answer must address the query, your answer can be textual, python code only, combination of both or you can even output several code snipets, if really necessary
- code snipets must be inside backticks! (the format is ```python\n<code>\n```)
- different code examples must be in separated code snipets (tj. if you are showing various examples to the user, you have to make ```python\n<code>\n``` block for each example)
- it is prefered to answer by a single code snipet (the code will be then executed by our application)
- When necessary, answer by a full textual output
- You should comment on your intentions and what you are trying to code, explain your thoughts.
- if the user's query is very clear in the sense of what opereations should be done over the given data, output just the code, without any additional text!

Adhere to these visualisation guidelines:
{seaborn_best_practices}

Regarding coding, there are these rules:
A code snipet must define a Python function named `def solve(df):` that fulfills the user's query and returns the result of the analysis.
Include all necessary import statements at the beginning of the code.
Use backticks to enclose the code.
Only include visualizations if the user's query explicitly requests them (e.g., plot, chart).
Do not test the function with commands like `print(solve(df))`; only define the function. Avoid including code descriptions or usage examples.
Ensure not to check the instance type against `int` or `float`; use `Number` instead.
Be cautious of data order—avoid changing it through operations like sorting.
Do not use `reset_index()`. Leave the DataFrame index unaltered after processing.
Before returning the result in the `solve` function, remove all auxiliary or temporary columns (e.g., a 'delta' column).
Be cautious of NaN, Inf and other special values - the table contains few of them.
Avoid using pd.DataFrame.reset_index() method, it may cause failure in our unittests.

Your answer:
""".format(seaborn_best_practices=seaborn_implementation)

    def append_to_code(self, extracted_code, n_dfs=None):
        if n_dfs is None or n_dfs <= 1:
            return extracted_code + "\n\n" + "result = solve(df)\nprint(result)\nplt.clf()"
        else:
            return extracted_code + "\n\n" + f"result = solve({', '.join([f'df_{i+1}' for i in range(n_dfs)])})\nplt.clf()"

    def formulate_result(self, code_segemnts_obj, full_answer):
        if code_segemnts_obj is None:
            return full_answer
        for i, code_segment_obj in enumerate(code_segemnts_obj):
            if "ERROR" in code_segment_obj['result_of_execution']:
                code_res = "ERROR, this code cannot be applied to your data."
            else:
                code_res = code_segment_obj['result_of_execution']
            extracted_code_origin = code_segemnts_obj[i]['generated_code'] + "\n```"
            extracted_code = code_segemnts_obj[i]['final_generated_code'] + "\n```"

            if len(full_answer) - len(extracted_code_origin) < 0.1 * len(full_answer) and not (len(extracted_code_origin) > 600) and len(code_segemnts_obj) == 1:
                return code_res
            full_answer = full_answer.replace(
                extracted_code_origin,
                extracted_code + "\nRESULT: {}".format(code_res)
            )
        return full_answer

    def map(self, code_segments, func, args):
        ls = list(map(lambda x: func(x, **args), code_segments))
        return ls

    def rename_plots(self, code, new_names):
        """
        This function replaces the save_plot_name in a given code line with new names.
        It can handle both plt.savefig and <plot_var>.savefig cases.

        Parameters:
        code_line (str): The line of code containing the savefig pattern.
        new_names (list): A list of new names to replace save_plot_name.

        Returns:
        str: The modified code line with save_plot_name replaced by new names.
        """
        # Regular expression pattern to match plt.savefig('{save_plot_name}') or <plot_var>.savefig('{save_plot_name}')
        pattern = r"(\b\w+?\.savefig\('.*?'\))"

        # Find all matches
        matches = list(re.finditer(pattern, code))

        # Check if the number of matches is equal to the number of new names
        if len(matches) != len(new_names):
            raise ValueError(
                "The number of new names must match the number of savefig calls in the code line.")

        # Replace each match with the corresponding new name
        for i, match in enumerate(matches):
            code = code[:match.start()] + re.sub(r"\('.*?'\)",
                                                 f"('{new_names[i]}')", match.group(0)) + code[match.end():]

        return code

    # LEGACY FUNCTIONS:

    def format_generate_steps_no_plot_prompt(self, *args, **kwargs):
        raise Exception(
            "This prompt strategy does not support generating steps.")

    def format_generate_steps_for_plot_save_prompt(self, *args, **kwargs):
        raise Exception(
            "This prompt strategy does not support generating steps.")

    def format_generate_steps_for_plot_show_prompt(self, *args, **kwargs):
        raise Exception(
            "This prompt strategy does not support generating steps.")

    def format_generate_code_prompt(self, head_number, df, user_query, plan, column_description, functions_description, use_gpt4o=False):
        return self.format('generate_code', df, user_query, plan, column_description, functions_description, head_number=head_number, use_gpt4o=use_gpt4o)

    def format_generate_code_for_plot_save_prompt(self, head_number, df, user_query, plan, column_description, functions_description, save_plot_name=""):
        return self.format('generate_code_for_plot_save', df, user_query, plan, column_description, functions_description, head_number=head_number, save_plot_name=save_plot_name)

    def format_generate_code_multiple_dfs_prompt(self, head_number, dfs, user_query, plan, column_descriptions, functions_description):
        return self.format("generate_code_multiple_dfs", dfs, user_query, plan, column_descriptions, functions_description, head_number=head_number)
