from .base_prompt_strategy import FunctionBasePromptStrategy

class PromptsCoderOnlyForFunctionGeneration(FunctionBasePromptStrategy):

    has_planner = False

    generate_code_gpt4_ = """You are highly proficient with Python and the pandas library. A user has submitted a query that you need to address: '{input}'. 
You also have a list of subtasks that need to be completed.

A pandas DataFrame named `df` has been provided. It has already been defined and populated with the necessary data, so there is no need to load or recreate it.
The output of `print(df.head({head_number}))` is:
{df_head}

Please pay close attention to the descriptions of the columns in the DataFrame:
{column_description}


Your task is to define a Python function named `def solve(df):` that fulfills the user's query and returns the result of the analysis.
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
"""

    generate_code = """You are really good with Python and the pandas library. The user provided a query that you need to help achieve: '{input}'. 
You also have a list of subtasks to be accomplished.

You have been presented with a pandas DataFrame named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}

{functions_description}

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
"""

    generate_code_multiple_dfs = """You are really good with Python and the pandas library. The user provided a query that you need to help achieve: '{input}'. 
You also have a list of subtasks to be accomplished.

You have been presented with several pandas DataFrames named `df_k`, where k is a number; e.g. `{df_examples}` etc.
The DataFrames `df_k` have already been defined and populated with the required data, so don't load it and don't create new ones.
Here are the first 5 rows of the given DataFrames, sorted (i.e. first 5 rows of `df_1`, then `df_2`, etc.):
{data}

{functions_description}

Return the definition of a Python function called `def solve({df_examples}):` that accomplishes the user query and returns the result of the analysis.
You must include the necessary import statements at the top of the code.
You must use the backticks to enclose the code.
Only show the visualizations if the user query explicitly asks for them (plot, chart, etc.).
All dataframes are already loaded, and defined. Do not define any other dataframe or table! Unused dataframes should be passed as optional arguments to the solve function.
Do not test the function with anything similar to `print(solve({df_examples}))`, only define the function, in the format of the following example:

Here are examples of the output format:

Example format:
```python
import pandas as pd

def solve({df_examples}):
    # Step 1: some comment.
    <CODE>

    # STEP 2: some comment.
    <CODE>

    # Step 3: some comment.
    <CODE>

    # Return the result.
    return <RESULT>
```
Your code: 
"""

    generate_code_multiple_dfs_plot = """You are really good with Python and the pandas library. The user provided a query that you need to help achieve: '{input}'. 
You also have a list of subtasks to be accomplished.

You have been presented with several pandas DataFrames named `df_k`, where k is a number; e.g. `{df_examples}` etc.
The DataFrames `df_k` have already been defined and populated with the required data, so don't load it and don't create new ones.
Here are the first 5 rows of the given DataFrames, sorted (i.e. first 5 rows of `df_1`, then `df_2`, etc.):
{data}

{functions_description}

Return the definition of a Python function called `def solve({df_examples}):` that accomplishes the user query and returns the result of the analysis.
You must include the necessary import statements at the top of the code.
You must use the backticks to enclose the code.
Only show the visualizations if the user query explicitly asks for them (plot, chart, etc.).
All dataframes are already loaded, and defined. Do not define any other dataframe or table! Unused dataframes should be passed as optional arguments to the solve function.
Do not test the function with anything similar to `print(solve({df_examples}))`, only define the function, in the format of the following example:
You must not include `plt.show()`. Just save the plot to '{save_plot_name}' with `plt.savefig('{save_plot_name}')`.

Here are examples of the output format:

Example format:
```python
import pandas as pd
import matplotlib.pyplot as plt

def solve({df_examples}):
    # Step 1: some comment.
    <CODE>

    # STEP 2: some comment.
    <CODE>

    # Step 3: some comment.
    <CODE>

    # Save the pie plot to 'plots/example_plot00.png'.
    save and return
```


"""

    generate_code_for_plot_save = """You are really good with Python and the pandas library. The user provided a query that you need to help achieve: '{input}'. 

You have been presented with a pandas DataFrame named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}

{functions_description}

Return the definition of a Python function called `def solve(df):` that accomplishes the user query and returns the result of the analysis.
You must include the necessary import statements at the top of the code.
You must not include `plt.show()`. Just save the plot to '{save_plot_name}' with `plt.savefig('{save_plot_name}')`.
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
"""

    def format_generate_steps_no_plot_prompt(self, *args, **kwargs):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_steps_for_plot_save_prompt(self, *args, **kwargs):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_steps_for_plot_show_prompt(self, *args, **kwargs):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_code_prompt(self, head_number, df, user_query, plan, column_description, functions_description, use_gpt4o=False):
        prompt = self.generate_code if use_gpt4o is False else self.generate_code_gpt4_
        return prompt.format(
            input=user_query, 
            df_head=df.head(head_number), 
            plan=plan,
            head_number=head_number, 
            column_description=column_description, 
            functions_description=functions_description
        )

    def format_generate_code_for_plot_save_prompt(self, head_number, df, user_query, plan, column_description, functions_description, save_plot_name=""):
        assert save_plot_name, "The save_plot_name parameter must be provided for this prompt strategy."

        return self.generate_code_for_plot_save.format(
            input=user_query, 
            df_head=df.head(head_number), 
            plan=plan,
            head_number=head_number, 
            column_description=column_description, 
            save_plot_name=save_plot_name, 
            functions_description=functions_description
        )
    
    def format_generate_code_multiple_dfs_prompt(self, head_number, dfs, user_query, plan, column_descriptions, functions_description):
        data = "\n"
        for i in range(len(dfs)):
            data += f"DataFrame df_{i+1}:\n{dfs[i].head(head_number)}\n{column_descriptions[i]}\n"

        return self.generate_code_multiple_dfs.format(
            input=user_query, 
            plan=plan,
            functions_description=functions_description,
            data=data,
            df_examples=", ".join([f"df_{i+1}" for i in range(len(dfs))])
        )
    
    def format_generate_code_multiple_dfs_plot_prompt(self, head_number, dfs, user_query, plan, column_descriptions, functions_description, save_plot_name=""):
        data = "\n"
        for i in range(len(dfs)):
            data += f"DataFrame df_{i+1}:\n{dfs[i].head(head_number)}\n{column_descriptions[i]}\n"

        print("MULTIPLE DFSSSSSSSS")
        print(", ".join([f"df_{i+1}" for i in range(len(dfs))]))
        return self.generate_code_multiple_dfs.format(
            input=user_query, 
            plan=plan,
            functions_description=functions_description,
            data=data,
            save_plot_name=save_plot_name, 
            df_examples=", ".join([f"df_{i+1}" for i in range(len(dfs))])
        )
