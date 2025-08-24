import json
import pandas as pd

from .logger import *
from .prompts_dir.function_generation_coder_only import PromptsCoderOnlyForFunctionGeneration
from .prompts_dir.simple_coder_only import PromptsSimpleCoderOnly
from .prompts_dir.simple import PromptsSimple
from .prompts_dir.function_generation import PromptsForFunctionGeneration
from .prompts_dir.debug_prompts import DebugPrompts
from .prompts_dir.hybrid_prompts import HybridPromptsForFunctionAndTextGeneration


class Prompts:
    def __init__(
        self,
        str_strategy: PromptsSimple | PromptsForFunctionGeneration,
        head_number: int,
        debug_strategy: str = "basic",
        functions_description: str = None
    ):
        """
        Class providing prompts for the LLM/Agent.

        Parameters
        ----------
        functions_description: str = None
            Description of the 'pre-cooked' functions which agent might decide to use in its code.
        """
        self.head_number = head_number

        if str_strategy == "functions":
            self.strategy = PromptsForFunctionGeneration()
        elif str_strategy == "simple":
            self.strategy = PromptsSimple()
        elif str_strategy == "coder_only_simple":
            self.strategy = PromptsSimpleCoderOnly()
        elif str_strategy == "coder_only_functions":
            self.strategy = PromptsCoderOnlyForFunctionGeneration()
        elif str_strategy == 'hybrid_code_text':
            self.strategy = HybridPromptsForFunctionAndTextGeneration()
        else:
            raise Exception(f"{RED}Unknown prompt strategy!{RESET}")

        if functions_description is not None:
            self.functions_description_for_plannig = f"""\
The code generation assistant will also have available several predefined functions. If you find them useful for some step you can include them in it.
These functions are automaticaly available for the code generation assistant in the code and you must not instruct it to import them.
The functions available are as follows:

{functions_description}
"""
            self.functions_description_for_coding = f"""\
In writing the code you have also have available several predefined functions.
They are already defined in your context and therefore are already available to you and you must not redefine them yourself.
When you are instructed to use a predefined function in any step of the plan you must just run it, you must never create or define it!
The descriptions of these functions go as follows:

{functions_description}
"""
        else:
            self.functions_description_for_plannig = ''
            self.functions_description_for_coding = ''

        debug_strats = {"basic": DebugPrompts.basic_debug_prompt,  # str
                        "completion": DebugPrompts.completion_debug_prompt}

        if debug_strategy in debug_strats:
            self.debug_strategy = debug_strats[debug_strategy]
        else:
            raise Exception(f"{RED}Unknown {CYAN}debug\
                            {RED} prompt strategy! See __init__ of Prompts() in prompts.py{RESET}")

    # Changes if we change the 'df' from outside. Creates description dynamically.
    def column_annotations(self, data_annotations: dict | list[dict] | None):
        if isinstance(data_annotations, dict):
            return f"Here is also a dump of JSON file describing columns of the dataframe 'df':\n {json.dumps(data_annotations, indent=4)}"
        elif isinstance(data_annotations, list):
            return 'Here is also a dump of JSON files describing columns of the dataframes:\n' + '\n'.join(f'df_{i + 1} {json.dumps(data_annotation, indent=4)}' for i, data_annotation in enumerate(data_annotations))
        else:
            return ''

    def generate_steps_no_plot_prompt(self, df: pd.DataFrame, user_query: str, data_annotation: dict | None):
        return self.strategy.format_generate_steps_no_plot_prompt(head_number=self.head_number, df=df, user_query=user_query, column_description=self.column_annotations(data_annotation))

    def generate_replan(self, df: pd.DataFrame, user_query: str, plan: str, data_annotation: dict | None):
        return self.strategy.format_reformulate_plan_prompt(self.head_number, df, user_query, self.column_annotations(data_annotation), plan, self.functions_description_for_plannig)

    def generate_steps_for_plot_save_prompt(self, df: pd.DataFrame, user_query: str, save_plot_name: str, data_annotation: dict | None):
        return self.strategy.format_generate_steps_for_plot_save_prompt(self.head_number, df, user_query, save_plot_name, self.column_annotations(data_annotation), self.functions_description_for_plannig)

    def generate_steps_for_plot_show_prompt(self, df: pd.DataFrame, user_query: str, data_annotation: dict | None):
        return self.strategy.format_generate_steps_for_plot_show_prompt(self.head_number, df, user_query, self.column_annotations(data_annotation), self.functions_description_for_plannig)

    def generate_code_prompt(self, df: pd.DataFrame, user_query: str, plan: str, data_annotation: dict | None):
        return self.strategy.format_generate_code_prompt(self.head_number, df, user_query, plan, self.column_annotations(data_annotation), self.functions_description_for_coding)

    def generate_code_multiple_dfs_prompt(self, dfs: list[pd.DataFrame], user_query: str, plan: str, data_annotation: dict | None | list[dict]):
        return self.strategy.format_generate_code_multiple_dfs_prompt(self.head_number, dfs, user_query, plan, self.column_annotations(data_annotation), self.functions_description_for_coding)

    def generate_code_for_plot_save_prompt(self, df: pd.DataFrame, user_query: str, plan: str, data_annotation: dict | None, save_plot_name=""):
        return self.strategy.format_generate_code_for_plot_save_prompt(self.head_number, df, user_query, plan, self.column_annotations(data_annotation), self.functions_description_for_coding, save_plot_name=save_plot_name)

    def generate_code_multiple_dfs_plot_prompt(self,  dfs: list[pd.DataFrame], user_query: str, plan: str, data_annotation: dict | None, save_plot_name=""):
        return self.strategy.format('generate_code_multiple_dfs_plot', dfs, user_query, plan, data_annotation, self.functions_description_for_coding, save_plot_name=save_plot_name, head_number=self.head_number)
    
    def query_rewrite(self, df: pd.DataFrame, user_query: str, data_annotation: dict | None):
        return self.strategy.format_query_rewrite(self.head_number, df, user_query, self.column_annotations(data_annotation), self.functions_description_for_plannig)

    def is_query_clear(self, df, user_query, data_annotation: dict | None):
        return self.strategy.format_is_query_clear(self.head_number, df, user_query, self.column_annotations(data_annotation), self.functions_description_for_plannig)

    def query_disambiguation(self, df, user_query, data_annotation: dict | None):
        return self.strategy.format_query_disambiguation(self.head_number, df, user_query, self.column_annotations(data_annotation), self.functions_description_for_plannig)

    def fix_code_prompt(self, df, user_query, code_to_be_fixed, error_message, initial_coder_prompt):
        return self.debug_strategy.format(code=code_to_be_fixed, error=error_message, input=user_query,
                                          df=df, initial_coder_prompt=initial_coder_prompt)

    def get_merge_query_and_history(self, query, history):
        return self.merge_query_and_history_prompt.format(chat_history=history, question=query)

    def get_answer_noncoding_query_prompt(self, query, history):
        return PromptsForSpecificFunctionality.answer_noncoding_query.format(chat_history=history, input=query, question=query)

    def get_save_df_prompt(self, query):
        return query + " " + PromptsForSpecificFunctionality.save_df

    def is_coding_needed_cls(self, df, user_query):
        return PromptsForSpecificFunctionality.is_coding_needed_cls.format(df=df.head(self.head_number), input=user_query)

    def _generate_whatever(self, df: pd.DataFrame, user_query: str, plan: str, data_annotation: dict | None):
        return self.strategy.format(
            'generate_whatever',
            df,
            user_query,
            plan,
            self.column_annotations(data_annotation),
            self.functions_description_for_coding,
            head_number=self.head_number
        )

    def get_prompt(self, prompt_name, *args, **kwargs):
        if prompt_name == 'generate_whatever':
            return self._generate_whatever(*args, **kwargs)
        elif prompt_name == 'generate_steps_no_plot':
            return self.generate_steps_no_plot_prompt(*args, **kwargs)
        elif prompt_name == 'generate_replan':
            return self.generate_replan(*args, **kwargs)
        elif prompt_name == 'generate_steps_for_plot_save':
            return self.generate_steps_for_plot_save_prompt(*args, **kwargs)
        elif prompt_name == 'generate_steps_for_plot_show':
            return self.generate_steps_for_plot_show_prompt(*args, **kwargs)
        elif prompt_name == 'generate_code':
            return self.generate_code_prompt(*args, **kwargs)
        elif prompt_name == 'generate_code_multiple_dfs':
            return self.generate_code_multiple_dfs_prompt(*args, **kwargs)
        elif prompt_name == 'generate_code_multiple_dfs_plot':
            return self.generate_code_multiple_dfs_plot_prompt(*args, **kwargs)
        elif prompt_name == 'generate_code_for_plot_save':
            return self.generate_code_for_plot_save_prompt(*args, **kwargs)
        elif prompt_name == 'query_rewrite':
            return self.query_rewrite(*args, **kwargs)
        elif prompt_name == 'is_query_clear':
            return self.is_query_clear(*args, **kwargs)
        elif prompt_name == 'query_disambiguation':
            return self.query_disambiguation(*args, **kwargs)
        elif prompt_name == 'fix_code':
            return self.fix_code_prompt(*args, **kwargs)
        elif prompt_name == 'merge_query_and_history':
            return self.get_merge_query_and_history(*args, **kwargs)
        elif prompt_name == 'answer_noncoding_query':
            return self.get_answer_noncoding_query_prompt(*args, **kwargs)
        elif prompt_name == 'save_df':
            return self.get_save_df_prompt(*args, **kwargs)
        elif prompt_name == 'is_coding_needed_cls':
            return self.is_coding_needed_cls(*args, **kwargs)
        else:
            raise ValueError(f"Unknown prompt name: {prompt_name}")


RESULT_INTERPRETER_PROMPT = """Role:
You are a specialized agent that interprets pandas DataFrame query results and determines if they adequately answer the user's question.
Input Analysis:
You will receive:

User Query: The original question asked by the user
Generated Code: The pandas code that was executed
Print Result: The output from running the code
Table Annotation: Description of the table structure and columns

Task:
Analyze these inputs and provide a structured response in the following JSON format:
{{
"interpretation": "string",
"rerun_needed": boolean,
"fixes": "string"
}}

Field Definitions:
interpretation (string):
CRITICAL: Always provide COMPLETE information from PRINT_RESULT. Never use "etc." or truncate results. The user needs all available data, not summaries.

Provide a clear, natural language explanation of what the results show
Clean and process ALL raw data silently - parse formatting, split concatenated values, remove control characters
Present ALL findings from the results - NEVER truncate or use "etc."
Use English language.
Include ALL relevant details:
- Complete list of all values found (never truncated)
- All specific cleaned values and patterns
- NEVER mention formatting issues or data quality problems to the user

If no results found, explain this clearly
Make it conversational and helpful - focus only on the clean, complete results
Silently handle all formatting issues: Split newlines/carriage returns, clean whitespace, separate multiple values - present the clean data without mentioning these corrections.

rerun_needed (boolean):
Set to true ONLY if these major issues apply:

Empty results: No data matches the query criteria due to filtering problems
Overly broad results: Thousands of results without sufficient filtering
Incomplete filtering: Code doesn't use important query criteria (missing year, make, model)
Wrong columns returned: Code returns completely wrong or insufficient columns for the query
Major logic errors: Filtering logic fundamentally doesn't match the user's intent
Structural data issues: Data structure problems that can't be resolved in interpretation
Wrong table chosen: From the table annotation it's claer that the wrong table was chosen for the user's query

Set to false if:

Results can be cleaned and interpreted (even if formatting is messy)
Data contains the answer to user's question
Minor formatting issues can be handled in interpretation
Appropriate filtering was applied

fixes (string):
If rerun_needed = true, provide specific, actionable instructions:

For empty results: Suggest relaxing criteria, checking for typos, or alternative search approaches
For major filtering problems: Identify which important criteria are missing or incorrect
For structural issues: Suggest fundamental changes to data extraction approach

If rerun_needed = false, leave this as an empty string: ""
Quality Guidelines:
Handle in Interpretation (Don't Rerun):

Formatting issues (\r\n, \t, excessive whitespace)
Multiple values in single cell that need separation
Minor data cleaning and parsing
Results that answer the question but need formatting

Rerun Only For:

Completely empty results due to bad filtering
Wrong data entirely (asked for alternators, got steering units)
Missing crucial search criteria
Fundamental logic errors in code

Important Notes:

Always consider the user's specific question when evaluating result adequacy
Pay attention to data quality and formatting issues
Be specific in your fixes - provide actual code suggestions when possible
Consider cultural context (e.g., respond in the same language as the query)
Focus on practical utility for the end user
Clean up formatting issues in the interpretation rather than requesting reruns

Here is the input data:
<user_query>
{USER_QUERY}
</user_query>

<generated_code>
{GENERATED_CODE}
</generated_code>

<print_result>
{PRINT_RESULT}
</print_result>

<table_annotation>
{TABLE_ANNOTATION}
</table_annotation>

Perform the analysis and complete the task. 
Return only the output json and nothing else.
"""

RESULT_INTERPRETER_PROMPT_2 = """Role:
You are a specialized agent that interprets pandas DataFrame query results and determines if they adequately answer the user's question.
Input Analysis:
You will receive:

User Query: The original question asked by the user
Generated Code: The pandas code that was executed
Print Result: The output from running the code
Table Annotation: Description of the table structure and columns

Task:
Analyze these inputs and provide a structured response in the following JSON format:
{{
"interpretation": "string",
"rerun_needed": boolean,
"fixes": "string"
}}

Field Definitions:
interpretation (string):

Provide a clear, natural language explanation of what the results show
Clean and process the raw data - parse messy formatting, split concatenated values, remove control characters
Reference specific findings from the processed results
Use English language
Include relevant details like:

Number of matches found
Specific cleaned values or patterns identified
Any data quality observations

If no results found, explain this clearly
Make it conversational and helpful
Handle common formatting issues directly: Split newlines/carriage returns, clean whitespace, separate multiple values

rerun_needed (boolean):
Set to true ONLY if these major issues apply:

Empty results: No data matches the query criteria due to filtering problems
Overly broad results: Thousands of results without sufficient filtering
Incomplete filtering: Code doesn't use important query criteria (missing year, make, model)
Wrong columns returned: Code returns completely wrong or insufficient columns for the query
Major logic errors: Filtering logic fundamentally doesn't match the user's intent
Structural data issues: Data structure problems that can't be resolved in interpretation
Wrong table chosen: From the table annotation it's claer that the wrong table was chosen for the user's query

Set to false if:

Results can be cleaned and interpreted (even if formatting is messy)
Data contains the answer to user's question
Minor formatting issues can be handled in interpretation
Appropriate filtering was applied

fixes (string):
If rerun_needed = true, provide specific, actionable instructions:

For empty results: Suggest relaxing criteria, checking for typos, or alternative search approaches
For major filtering problems: Identify which important criteria are missing or incorrect
For structural issues: Suggest fundamental changes to data extraction approach

If rerun_needed = false, leave this as an empty string: ""
Quality Guidelines:
Handle in Interpretation (Don't Rerun):

Formatting issues (\r\n, \t, excessive whitespace)
Multiple values in single cell that need separation
Minor data cleaning and parsing
Results that answer the question but need formatting

Rerun Only For:

Completely empty results due to bad filtering
Wrong data entirely (asked for alternators, got steering units)
Missing crucial search criteria
Fundamental logic errors in code

Example Response Format:
For a result with formatting issues (handled in interpretation):
{
"interpretation": "Найдены альтернаторы для Volvo XC40 2017 года. OEM номера: 0127813035, 36003710, 31652870, 32300720 (первый альтернатор) и 32137503 (второй альтернатор). Это оригинальные номера производителя, которые можно использовать для заказа или кросс-референса.",
"rerun_needed": false,
"fixes": ""
}
For a result needing major fixes:
{
"interpretation": "The search criteria were applied but returned no results. This could mean no alternators are available for this specific vehicle configuration in the database.",
"rerun_needed": true,
"fixes": "Try broadening the search - remove the year filter first (just 'volvo' and 'xc40'), or check if the model name might be stored differently (like 'XC-40' or 'XC 40')."
}
Important Notes:

Always consider the user's specific question when evaluating result adequacy
Pay attention to data quality and formatting issues
Be specific in your fixes - provide actual code suggestions when possible
Consider cultural context (e.g., respond in the same language as the query)
Focus on practical utility for the end user
Clean up formatting issues in the interpretation rather than requesting reruns

Here is the input data:
<user_query>
{USER_QUERY}
</user_query>

<generated_code>
{GENERATED_CODE}
</generated_code>

<print_result>
{PRINT_RESULT}
</print_result>

<table_annotation>
{TABLE_ANNOTATION}
</table_annotation>

Perform the analysis and complete the task. 
Return only the output json and nothing else.
"""