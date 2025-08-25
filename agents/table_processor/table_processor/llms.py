from typing import Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Callable
import os
import json
import pandas as pd
import time
from enum import Enum
from .prompts import *
from dotenv import load_dotenv
import logging
import sys
sys.path.append("../..")  # to import config from base folder

# Color constants for terminal output
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


# *USE THIS* Start
logger = logging.getLogger(__name__)
load_dotenv()
api_key = os.environ.get("OPENAI_TOKEN")
client = OpenAI(api_key=api_key)

class TopicClassifier(BaseModel):
    """Classify if the user asked for a vizualization, e.g. plot or graph, or asked for some general numerical result, e.g. finding correlation or maximum value."""

    topic: Literal["plot", "general"]
    "The topic of the user question. One of 'plot' or 'general'."


class Tagging(BaseModel):
    """Tag the piece of text with particular info. Classify if the user asked for a vizualization, e.g. plot or graph, or asked for some general numerical result, e.g. finding correlation or maximum value."""
    topic: str = Field(
        description="Topic of user's query, must be `plot` or `general`.")


class Role(Enum):
    PLANNER = 0
    CODER = 1
    DEBUGGER = 2


class LLM:

    def __init__(
        self,
        llm_client: OpenAI,
        llm_model = 'gpt-5-2025-08-07',
        head_number=2,
        prompt_strategy="simple",
        functions_description: str|None = None,
        custom_client=None
    ):
        """
        Creates LLM wrapper using model `model`.

        Parameters
        ----------
        functions_description: str = None
            Description of the 'pre-cooked' functions which agent might decide to use in its code.
        custom_client: OpenAI client instance = None
            Pre-configured OpenAI client with custom parameters
        """
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.client = custom_client if custom_client else client  # Use custom client or default
        self.head_number = head_number
        self.local_coder_model = None
        self.conversation_history = []  # Store conversation history

        self.prompts = Prompts(str_strategy=prompt_strategy,
                               head_number=head_number, functions_description=functions_description)
        self._call_openai_llm = self.call_llm
    
    def call_llm(self, prompt:str):
        response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                # max_tokens=16200,
                # temperature=0.01
            )
        return response.choices[0].message.content
    
    def generate_column_descriptions(self, table_name: str, columns: list) -> str:
        """
        Generates concise descriptions for columns based on their names and types,
        and includes units if available.
        """
        descriptions_dict = []

        context = "Context of the table:\n"
        for col in columns:
            context += f"- {col['name']} (Type: {col['type']})\n"

        start_time = time.time()

        for column in columns:
            # column_start_time = time.time()
            combined_prompt = (
                f"Generate a brief and simple description for the column '{column['name']}' "
                f"without mentioning the column name or table name. "
                f"Focus on what the column represents. Also, identify and specify the most likely unit of measurement for the column "
                f"such as km for distance, km/h for speed, percentage for ratios, degrees Celsius for temperature, or amperes for current, volts for voltage, kW for power and kWh for energy. "
                f"If the column does not have a specific unit, return 'None'."
                f"Return the description and the unit in the format: Description: <description>, Unit: <unit>.\n\n"
                f"{context}\n"
                f"Description and unit for the column '{column['name']}':"
            )

            combined_result = self.call_llm(combined_prompt)[0]

            # Extract description and unit from the combined result
            try:
                description, unit = combined_result.strip().split('Unit:')
                description = description.replace('Description:', '').strip()
                unit = unit.strip()
                if unit.lower() in ['none', '', 'no units']:
                    unit = None
            except ValueError:
                # Handle case where the unit is not provided
                description = combined_result.strip()
                unit = None

            print(f'{column["name"]}: {unit}')

            descriptions_dict.append({
                "name": column['name'],
                "type": column['type'],
                "unit": unit,
                "description": description
            })

            # column_end_time = time.time()
            # print(f"Time taken for column '{column['name']}': {column_end_time - column_start_time:.2f} seconds")

        end_time = time.time()
        print(f"Total time taken for all columns: {end_time - start_time:.2f} seconds")

        return json.dumps(descriptions_dict)

    @staticmethod
    def structured_decoding(user_query: str):
        pass

    def get_prompt_from_router(self, user_query, df, data_annotation):
        """
        Select a prompt between the one saving the plot and the one calculating some math.
        """
        print("SELECTING A PROMPT")
        
        template = self.prompts.generate_steps_no_plot_prompt(df, user_query, data_annotation)
            
        return template

    def plan_steps_with_gpt(self, user_query, df, data_annotation: dict | None = None):
        temlate_for_math_planner = self.prompts.generate_steps_no_plot_prompt(
            df, user_query, data_annotation)

        selected_prompt = temlate_for_math_planner

        return self.call_llm(selected_prompt)

    def query_rewrite(self, user_query, df, save_plot_name=None, query_type=None, data_annotation: dict | None = None):

        template_prompt = self.prompts.query_rewrite(
            df, user_query, data_annotation)

        return self._call_openai_llm(template_prompt), template_prompt

    def generate_replan(self, user_query, df, plan, save_plot_name=None, query_type=None, data_annotation: dict | None = None):
        template_prompt = self.prompts.generate_replan(
            df, user_query, plan, data_annotation)
        return self._call_openai_llm(template_prompt)

    def merge_query_and_history(self, user_query, history):
        template_prompt = self.prompts.get_merge_query_and_history(
            user_query, history)
        return self._call_openai_llm(template_prompt)

    def is_query_clear(self, user_query, df, data_annotation: dict | None = None):
        template_prompt = self.prompts.is_query_clear(
            df, user_query, data_annotation)
        return self._call_openai_llm(template_prompt)

    def is_coding_needed_cls(self, user_query, df) -> int:
        # 0 - no coding required, 1 - coding required
        template_prompt = self.prompts.is_coding_needed_cls(df, user_query)
        answer = self._call_openai_llm(template_prompt)
        print(answer)
        if '1' in answer:
            return 1
        return 0

    def answer_noncoding_query(self, user_query, chat_history):
        template_prompt = self.prompts.get_answer_noncoding_query_prompt(
            user_query, chat_history)
        return self._call_openai_llm(template_prompt)

    def modify_query_for_save_df(self, user_query):
        return self.prompts.get_save_df_prompt(user_query)

    def query_disambiguation(self, user_query, df, data_annotation: dict | None = None):
        template_prompt = self.prompts.query_disambiguation(
            df, user_query, data_annotation)
        return self._call_openai_llm(template_prompt)

    def generate_code(self,
                      user_query,
                      df,
                      plan,
                      show_plot=False,
                      tagged_query_type="general",
                      llm="gpt-3.5-turbo-1106",
                      adapter_path="",
                      save_plot_name="",  # for the "coder_only" prompt strategies
                      data_annotation: dict | None = None,
                      prompt_name: str | None = None
                      ):
        if isinstance(df, pd.DataFrame) and not tagged_query_type == "plot":
            # instruction_prompt = self.prompts.generate_code_prompt(df, user_query, plan, data_annotation)
            instruction_prompt = self.prompts.get_prompt(
                prompt_name, df, user_query, plan, data_annotation)
        elif isinstance(df, list) and not tagged_query_type == "plot":
            instruction_prompt = self.prompts.generate_code_multiple_dfs_prompt(
                df, user_query, plan, data_annotation)
        # don't include plt.show() in the generated code
        elif isinstance(df, pd.DataFrame) and tagged_query_type == "plot" and not show_plot:
            instruction_prompt = self.prompts.generate_code_for_plot_save_prompt(
                df, user_query, plan, data_annotation, save_plot_name=save_plot_name)
        elif isinstance(df, list) and tagged_query_type == 'plot':
            instruction_prompt = self.prompts.generate_code_multiple_dfs_plot_prompt(
                df, user_query, plan, data_annotation, save_plot_name=save_plot_name)
        else:
            # TODO: implement for multiple dfs and tagged_query_type == "plot" and show_plot=True
            instruction_prompt = self.prompts.generate_code_for_plot_save_prompt(
                df[-1], user_query, plan, data_annotation, save_plot_name=save_plot_name)

        if llm.startswith("gpt"):
            return self.call_llm(instruction_prompt)


    def fix_generated_code(self, df, code_to_be_fixed, error_message, user_query, initial_coder_prompt, data_annotation: dict | None = None):
        prompt = self.prompts.fix_code_prompt(
            df, user_query, code_to_be_fixed, error_message, initial_coder_prompt)
        return self._call_openai_llm(prompt), prompt

    def clear_conversation_history(self):
        """Clear the conversation history to start fresh."""
        self.conversation_history = []
        print("Conversation history cleared")
        
    def set_system_message(self, system_message: str):
        """Set a system message for the conversation context."""
        # Clear existing history and add system message
        self.conversation_history = [{"role": "system", "content": system_message}]
        print(f"System message set: {system_message[:50]}...")
    
    def generate_title(self, prompt: str) -> str:
        try:
            response = self._call_openai_llm(prompt)[0]
            title = response.strip() 
            return title
        except Exception as e:
            print(f"Error generating title: {e}")
            return "Untitled Conversation"