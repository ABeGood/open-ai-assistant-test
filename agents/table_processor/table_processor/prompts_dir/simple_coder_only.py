from .plotting_guidelines import seaborn_implementation
from .base_prompt_strategy import BasePromptStrategy

class PromptsSimpleCoderOnly(BasePromptStrategy):

    has_planner = False

    generate_code = """The user provided a query that you need to help achieve: '{input}'.

You have been presented with a pandas DataFrame named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}

{functions_description}

Return only the Python code that accomplishes the user query.
You must include the necessary import statements at the top of the code.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.
Only show the visualizations if the user query explicitly asks for them (plot, chart, etc.).

Example of the output format with backticks:
```python

```
"""

    generate_code_multiple_dfs = """The user provided a query that you need to help achieve: '{input}'.

You have been presented with several pandas DataFrames named `df_k`, where k is a number; e.g. `{df_examples}` etc.
The DataFrames `df_k` have already been defined and populated with the required data, so don't load it and don't create new ones.
Here are the first 5 rows of the given DataFrames, sorted (i.e. first 5 rows of `df_1`, then `df_2`, etc.):
{data}

{functions_description}

Return only the Python code that accomplishes the user query.
You must include the necessary import statements at the top of the code.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.
Only show the visualizations if the user query explicitly asks for them (plot, chart, etc.).
Use only the provided DataFrames {df_examples} in your code.

Example of the output format with backticks:
```python

```
"""

    generate_code_for_plot_save = """The user provided a query that you need to help achieve: '{input}'.

You have been presented with a pandas DataFrame named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}
{column_description}

{functions_description}

Return only the Python code that accomplishes the user query.
You must include the necessary import statements at the top of the code.
You must not include `plt.show()`. Just save the plot to '{save_plot_name}' with `plt.savefig('{save_plot_name}')`.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.
Only show the visualizations if the user query explicitly asks for them (plot, chart, etc.).
Use only the provided DataFrames {df_examples} in your code.

Example of the output format with backticks:
```python

```
"""

    def format_generate_steps_no_plot_prompt(self, *args, **kwargs):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_steps_for_plot_save_prompt(self, *args, **kwargs):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_steps_for_plot_show_prompt(self, *args, **kwargs):
        raise Exception("This prompt strategy does not support generating steps.")

    def format_generate_code_prompt(self, head_number, df, user_query, plan, column_description):
        return self.generate_code.format(
            input=user_query, 
            df_head=df.head(head_number), 
            plan=plan, head_number=head_number, 
            column_description=column_description, 
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
