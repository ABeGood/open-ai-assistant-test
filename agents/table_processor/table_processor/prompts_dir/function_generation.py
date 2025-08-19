from .plotting_guidelines import seaborn_implementation
from .base_prompt_strategy import FunctionBasePromptStrategy

class PromptsForFunctionGeneration(FunctionBasePromptStrategy):

    has_planner = True

    generate_steps_no_plot = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Formulate your response as an algorithm, breaking the solution into steps, including values necessary to answer the question, such as values to search for, and, most importantly, names of DataFrame columns.

This algorithm will later be used to write a Python function that takes an existing pandas DataFrame 'df' as an argument and returns the result of the analysis. 
The DataFrame 'df' is already defined and populated with necessary data. So there is no need to define it again or load it. Here's how the beginning of the 'df' looks: 
{df_head}
{column_description}

{functions_description}

Present your algorithm in no more than six simple, clear English steps. 
Focus on explaining the steps rather than writing code. 
Do not include any visualization steps, such as plots or charts. 
You must only output these steps; the code generation assistant will follow them to implement the solution. 
Finally, you must specify the precise value that the function should return.
Preferably, also state the Python data type of the result (e.g. float, DataFrame, list, string, dict, etc.).

Here are examples for your inspiration:

User question: 'What is the maximal voltage minus the minimal speed, but raised to the power of 3?'
Your output:
1. Find and store the minimal value in the 'Speed' column.
2. Find and store the maximal value in the 'Voltage' column.
3. Subtract the minimal speed from the maximal voltage.
4. Raise the result to the power of 3.
5. Return the resulting number.

User question: 'Find four car ids with the largest mileage'
Your output:
1. Sort the DataFrame `df` in descending order based on the 'Mileage' column.
2. Extract the first 4 rows from the sorted DataFrame.
3. Make a list of the 'car_id' column from the extracted DataFrame.
4. Return the list of car ids.
"""


    generate_steps_for_plot_save = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{{input}}'.

Formulate your response as an algorithm, breaking the solution into steps, including values necessary to answer the question, such as values to search for and, most importantly, names of DataFrame columns.
Make sure to state saving the plot to '{{plotname}}' in the last step. Do not include showing the plot to the user interactively; only save it to the '{{plotname}}'.

This algorithm will later be used to write Python code and applied to the existing pandas DataFrame `df`. 
The DataFrame `df` is already defined and populated with necessary data. So there is no need to define it again or load it. Here's the beginning of the 'df': 
{{df_head}}
{{column_description}}

{{functions_description}}

Present your algorithm in no more than six simple, clear English steps. 
Focus on explaining the steps rather than writing code. 
You must only output these steps; the code generation assistant will follow them to implement the solution.

Present your algorithm in up to six simple, clear English steps. Remember to explain steps rather than to write code. Implement Seaborn for plotting to adhere to the following enhanced plotting best practices:
{seaborn_implementation}

Make sure each plot includes:
- A clear title that reflects the content or the aim of the plot.
- Descriptive labels for the x and y axes, including units if applicable.

Here's an example for you:


User question: 'Create a bar plot of worst acceleration cars with voltage on the x axis and speed multiplied by 3 on the y axis'
Your output:
1. Sort the DataFrame `df` in descending order based on the 'Acceleration' column.
2. Extract the first 5 rows from the sorted DataFrame.
3. Multiply each 'Speed' value in the extracted DataFrame by 3.
4. Create a bar plot with 'Voltage' on the x axis and multiplied 'Speed' on the y axis.
5. Save the bar plot to 'plots/example_plot00.png'.
""".format(seaborn_implementation=seaborn_implementation)

    generate_steps_for_plot_show = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{{input}}'.

Formulate your response as an algorithm, breaking the solution into steps, including values necessary to answer the question, such as values to search for and, most importantly, names of DataFrame columns.
Make sure to state showing the plot in the last step.

This algorithm will later be used to write Python code and applied to the existing pandas DataFrame `df`. 
The DataFrame `df` is already defined and populated with necessary data. So there is no need to define it again or load it. Here's the beginning of the 'df': 
{{df_head}}
{{column_description}}

{{functions_description}}

Present your algorithm in up to six simple, clear English steps. 
Remember to explain steps rather than to write code.
You must output only these steps, the code generation assistant is going to follow them.

Present your algorithm in up to six simple, clear English steps. Remember to explain steps rather than to write code. Implement Seaborn for plotting to adhere to the following enhanced plotting best practices:
{seaborn_implementation}

Make sure each plot includes:
- A clear title that reflects the content or the aim of the plot.
- Descriptive labels for the x and y axes, including units if applicable.

Here's an example for your inspiration:
User question: 'Create a bar plot of worst acceleration cars with voltage on the x axis and speed multiplied by 3 on the y axis'
Your output:
1. Sort the DataFrame `df` in descending order based on the 'Acceleration' column.
2. Extract the first 5 rows from the sorted DataFrame.
3. Multiply each 'Speed' value in the extracted DataFrame by 3.
4. Create a bar plot with 'Voltage' on the x axis and multiplied 'Speed' on the y axis.
5. Show the plot.
""".format(seaborn_implementation=seaborn_implementation)

    generate_code = """You are really good with Python and the pandas library. The user provided a query that you need to help achieve: '{input}'. 
You also have a list of subtasks to be accomplished.

You have been presented with a pandas DataFrame named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}

Return the definition of a Python function called `def solve(df):` that accomplishes the following tasks and returns the result of the analysis if needed:
{plan}

{functions_description}

Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the necessary import statements at the top of the code.
You must use the backticks to enclose the code.
Do not test the function with anything similar to `print(solve(df))`, only define the function, like in the following examples:

Here are examples of the output format:

Example 1:
```python
import pandas as pd

def solve(df):
    # 1. Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
    <CODE>
    
    # 2. Extract the 5 rows from the sorted DataFrame.
    <CODE>
    
    # 3. Create a list of the 'Country' column from the extracted DataFrame.
    <CODE>
    
    # 4. Return the list of countries.
    return <RESULT>
```

Example 2:
```python
import pandas as pd
import random

def solve(df):
    # 1. Find and store the minimal value in the 'Speed' column.
    <CODE>
    
    # 2. Find and store the maximal value in the 'Voltage' column.
    <CODE>
    
    # 3. Subtract the minimal speed from the maximal voltage.
    <CODE>
    
    # 4. Raise the result to the random power.
    <CODE>
    
    # 5. Return the resulting number.
    return <RESULT>
```
"""

    generate_code_multiple_dfs = """You are really good with Python and the pandas library. The user provided a query that you need to help achieve: '{input}'. 
You also have a list of subtasks to be accomplished.

You have been presented with several pandas DataFrames named `df_k`, where k is a number; e.g. `{df_examples}` etc.
The DataFrames `df_k` have already been defined and populated with the required data, so don't load it and don't create new ones.
Here are the first 5 rows of the given DataFrames, sorted (i.e. first 5 rows of `df_1`, then `df_2`, etc.):
{data}

Return only the python code that accomplishes the following tasks:
{plan}

{functions_description}

Return the definition of a Python function called `def solve({df_examples}):` that accomplishes the user query and returns the result of the analysis.
Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
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
"""

    generate_code_for_plot_save = """You are really good with Python and the pandas library. The user provided a query that you need to help achieve: '{input}'. 
You also have a list of subtasks to be accomplished.

You have been presented with a pandas DataFrame named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}
Return the definition of a Python function called `def solve(df):` that accomplishes the following tasks:
{plan}

{functions_description}

Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the necessary import statements at the top of the code.
You must not include `plt.show()`. Just save the plot the way it is stated in the tasks.
You must use the backticks to enclose the code.
Do not test the function with anything similar to `print(solve(df))`, only define the function, like in the following examples:

Here is an example of the output format:
```python
import pandas as pd
import matplotlib.pyplot as plt

def solve(df):
    # 1. Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
    <CODE>
    
    # 2. Extract the 5 rows from the sorted DataFrame.
    <CODE>
    
    # 3. Create a pie plot of the 'GDP' column of the extracted DataFrame.
    <CODE>
    
    # 4. Save the pie plot to 'plots/example_plot00.png'.
    <CODE>
```
"""

    # Here, same as in SimplePrompts, but later could, for example, add column description prompts and methods would be different maybe
    def format_generate_steps_no_plot_prompt(self, head_number, df, user_query, column_description, functions_description):
        return self.generate_steps_no_plot.format(
            df_head=df.head(head_number), 
            input=user_query, 
            column_description=column_description, 
            functions_description=functions_description
        )

    def format_generate_steps_for_plot_save_prompt(self, head_number, df, user_query, save_plot_name, column_description, functions_description):
        return self.generate_steps_for_plot_save.format(
            input=user_query, 
            plotname=save_plot_name,
            df_head=df.head(head_number), 
            column_description=column_description, 
            functions_description=functions_description
        )

    def format_generate_steps_for_plot_show_prompt(self, head_number, df, user_query, column_description, functions_description):
        return self.generate_steps_for_plot_show.format(
            input=user_query, 
            df_head=df.head(head_number), 
            column_description=column_description, 
            functions_description=functions_description
        )

    def format_generate_code_prompt(self, head_number, df, user_query, plan, column_description, functions_description):
        return self.generate_code.format(
            input=user_query, 
            df_head=df.head(head_number), 
            plan=plan,
            head_number=head_number, 
            column_description=column_description, 
            functions_description=functions_description
        )

    def format_generate_code_for_plot_save_prompt(self, head_number, df, user_query, plan, column_description, functions_description, save_plot_name=""):
        return self.generate_code_for_plot_save.format(
            input=user_query, 
            df_head=df.head(head_number), 
            plan=plan,
            head_number=head_number, 
            column_description=column_description, 
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
