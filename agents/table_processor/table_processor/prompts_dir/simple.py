import pandas as pd
from .plotting_guidelines import plotting_best_practices, seaborn_implementation
from .base_prompt_strategy import BasePromptStrategy

class PromptsSimple(BasePromptStrategy):
    has_planner = True
    max_debug_times = 2

    generate_steps_no_plot = """You are an AI data analyst tasked with creating a plan to generate Python code for processing a pandas DataFrame. Your job is to assist the user with simple data analysis by formulating an algorithm that breaks down the solution into clear steps.

The user has asked the following question:
<input_question>
{INPUT_QUESTION}
</input_question>

The DataFrame 'df' is already defined and populated with necessary data. Here's the beginning of the 'df':
<dataframe_head>
{DATAFRAME_HEAD}
</dataframe_head>

To provide more context, here's the table annotation describing the meaning of each column:
<table_annotation>
{TABLE_ANNOTATION}
</table_annotation>

Your task is to create an algorithm that outlines the steps needed to answer the user's question using the provided DataFrame. This algorithm will later be used to write Python code and applied to the existing pandas DataFrame 'df'.

When formulating your response:
1. Break the solution into steps, including any values necessary to answer the question, such as names of DataFrame columns.
2. Be cautious of value order - sorting can change the order of rows, which may affect the results.
3. If a list of specific types is required, ensure these types are unique.
4. Present your algorithm with at most six simple, clear English steps.
5. Explain steps rather than writing code.
6. Don't include any visualization steps like plots or charts.

Here's an example of the expected output format:
1. Find and store the minimal value in the 'Speed' column.
2. Find and store the maximal value in the 'Voltage' column.
3. Subtract the minimal speed from the maximal voltage.
4. Raise the result to the third power.
5. Print the result.

Now, based on the user's question and the provided DataFrame information, please create and output your algorithm steps. Remember to focus on creating a plan that can be easily translated into Python code for processing the pandas DataFrame in the next step.
"""

    reformulate_plan = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

We have an algorithm (plan) that will later be used to write Python code and applied to the existing pandas DataFrame 'df'. 
The DataFrame 'df' is already defined and populated with necessary data. So there is no need to define it again or load it. Here's the beginning of the 'df': 
{df_head}
{column_description}

{functions_description}

Make sure, the plan is really precise and correct. If yes, output just the plan. Else modify the plan. Especially pay attention to whether the plan
will return the correct result, whether all steps are necessary and whether the estimated value is as expected from the user query. Beware
of values order - e.g. sort might change the order, hence two rows with equeal values might have different order after sort (i.e. sorting can be dangerous!, 
try to use other build in functions than sort).

The plan: 
{plan}

Tuned plan:
"""

    # TODO: {input} to ''
    generate_code = """The user provided a query that you need to help achieving: {input}. 
You also have a list of subtasks to be accomplished using Python.

Beware of values order - e.g. sort might change the order, hence two rows with equeal values might have different order after sort (i.e. it is not safe to use sort sometimes).

You have been presented with a pandas dataframe named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}

Return only the python code that accomplishes the following tasks:
{plan}

{functions_description}

Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the neccessery import statements at the top of the code.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.

df is already created

Example of the output format:
```python

```"""

    generate_code_multiple_dfs = """The user provided a query that you need to help achieving: {input}. 
You also have a list of subtasks to be accomplished using Python.

Beware of values order - e.g. sort might change the order, hence two rows with equeal values might have different order after sort (i.e. it is not safe to use sort sometimes).

You have been presented with several pandas DataFrames named `df_k`, where k is a number; e.g. `{df_examples}` etc.
The DataFrames `df_k` have already been defined and populated with the required data, so don't load it and don't create new ones.
Here are the first 5 rows of the given DataFrames, sorted (i.e. first 5 rows of `df_1`, then `df_2`, etc.):
{data}

Return only the python code that accomplishes the following tasks:
{plan}

{functions_description}

Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the neccessery import statements at the top of the code.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.
Use only the provided DataFrames {df_examples} in your code.
The DataFrames `df_k` have already been defined and populated with the required data, so don't load it and don't create new ones.
Note again, there is no 'df' DataFrame, only `df_k`, where k is a number.

Example of the output format:
```python

```
"""

    # TODO: {input} to ''
    fix_code = """You are a helpful assistant that corrects the python code that resulted in an error and returns the corrected code.

The code was designed to achieve this user request: {input}.
The DataFrame `df`, that we are working with has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}

The execution of the following code that was provided in the previous step resulted in an error:
```python
{code}
```

The error message is: '{error}'

{functions_description}

Return a corrected python code that fixes the error.
Always include the import statements at the top of the code, and comments and print statements where necessary.
Use the same format with backticks. Example of the output format:
```python

```"""

    query_rewrite = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Reformulate the given question into a query (in natural language) s.t.: the query is not too wordy and more precise. It must be clear from the query,
what analytical steps to do, hence the query should be easily tranlastable to SQL or other similar language. It is extremely crucial, that the
query must ask exactly the same thing as the original question.

Just for info, the question is intended for the following table defined in pandas: 
{df_head}
{column_description}

{functions_description}

Do not change anything when not necessary.

"""


    def format_generate_steps_no_plot_prompt(self, head_number, df, user_query, column_description):
        return self.generate_steps_no_plot.format(
            DATAFRAME_HEAD=df.head(head_number), 
            INPUT_QUESTION=user_query, 
            TABLE_ANNOTATION=column_description, 
        )
    
    def format_reformulate_plan_prompt(self, head_number, df, user_query, plan, column_description, functions_description):
        return self.generate_steps_no_plot.format(
            df_head=self.get_df_heads_str(df, head_number), 
            input=user_query, 
            column_description=column_description, 
            plan=plan, 
            functions_description=functions_description
        )

    def format_generate_code_prompt(self, head_number, df, user_query, plan, column_description, functions_description):
        return self.generate_code.format(
            input=user_query, 
            df_head=self.get_df_heads_str(df, head_number), 
            plan=plan, 
            head_number=head_number, 
            column_description=column_description, 
            functions_description=functions_description
        )

    def format_fix_code_prompt(self, head_number, df, user_query, code_to_be_fixed, error_message, column_description, functions_description):
        return self.fix_code.format(
            code=code_to_be_fixed, 
            df_head=self.get_df_heads_str(df, head_number), 
            error=error_message, 
            input=user_query, 
            head_number=head_number, 
            column_description=column_description, 
            functions_description=functions_description
        )
    
    def format_query_rewrite(self, head_number, df, user_query, column_description, functions_description):
        return self.query_rewrite.format(
            df_head=self.get_df_heads_str(df, head_number), 
            input=user_query, 
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
