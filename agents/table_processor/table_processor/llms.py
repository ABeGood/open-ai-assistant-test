import agents.config as config
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from typing import Literal
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser, JsonOutputFunctionsParser
from langchain.pydantic_v1 import BaseModel, Field
# from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_function
from operator import itemgetter
from openai import OpenAI, AzureOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import AzureChatOpenAI

import time
from enum import Enum

from .prompts import *
from .coder_llms import *

import sys
sys.path.append("../..")  # to import config from base folder


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
        model="gpt-3.5-turbo-1106",
        use_assistants_api=False,
        head_number=2,
        prompt_strategy="simple",
        functions_description: str = None
    ):
        """
        Creates LLM wrapper using model `model`.

        Parameters
        ----------
        functions_description: str = None
            Description of the 'pre-cooked' functions which agent might decide to use in its code.
        """
        self.model = model
        self.head_number = head_number
        self.local_coder_model = None

        self.prompts = Prompts(str_strategy=prompt_strategy,
                               head_number=head_number, functions_description=functions_description)
        if use_assistants_api:
            self._call_openai_llm = self._get_response_from_assistant
            self.my_assistant = self.model.beta.assistants.create(
                name="Data Assistant",
                instructions="""You are a data analyst who can analyse and interpret data files such as CSV and XLSX and can draw conclusions from the data. """ +
                """If user asks for a plot or for any other sort of visualization, just return a png file with what he asked for.""",  # must state to return a file
                tools=[{"type": "code_interpreter"}],
            )
            self.thread = self.model.beta.threads.create()
        else:
            self._call_openai_llm = self._get_response_from_base_gpt

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

            combined_message = [HumanMessage(content=combined_prompt)]
            combined_response = self.model.invoke(combined_message)

            # Check if combined response is a list and extract the content accordingly
            if isinstance(combined_response, list) and len(combined_response) > 0:
                combined_result = combined_response[0]['content']
            elif hasattr(combined_response, 'content'):
                combined_result = combined_response.content
            else:
                raise ValueError("Unexpected response format")

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

    def tag_query_type(self, user_query: str):
        """
        Select a prompt between the one saving the plot and the one calculating some math.
        """
        tagging_functions = [convert_to_openai_function(Tagging)]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Think carefully, and then tag the text as instructed"),
            ("user", "{input}")
        ])
        model_with_functions = self.model.bind(
            functions=tagging_functions,
            function_call={"name": "Tagging"}
        )
        tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()

        # If halts => could be a problem with parser
        query_topic = tagging_chain.invoke({"input": user_query})["topic"]

        print(f"{BLUE}SELECTED A PROMPT:{RESET} {YELLOW}{query_topic}{RESET}")

        return query_topic

    # Halts on chain.invoke() sometimes
    def get_prompt_from_router(self, user_query, df, save_plot_name=None):
        """
        Select a prompt between the one saving the plot and the one calculating some math.
        """
        print(f"{BLUE}SELECTING A PROMPT:{RESET}")
        temlate_for_plot_planner = self.prompts.generate_steps_for_plot_show_prompt(
            df, user_query)
        if save_plot_name is not None:
            temlate_for_plot_planner = self.prompts.generate_steps_for_plot_save_prompt(
                df, user_query, save_plot_name)
        temlate_for_math_planner = self.prompts.generate_steps_no_plot_prompt(
            df, user_query)

        prompt_branch = RunnableBranch(
            (lambda x: x["topic"] == "plot", PromptTemplate.from_template(
                temlate_for_plot_planner)),
            (lambda x: x["topic"] == "general",
             PromptTemplate.from_template(temlate_for_math_planner)),
            # default branch (must be included)
            PromptTemplate.from_template(temlate_for_math_planner)
        )

        classifier_function = convert_to_openai_function(TopicClassifier)
        llm = self.model.bind(
            functions=[classifier_function], function_call={
                "name": "TopicClassifier"}
        )
        parser = PydanticAttrOutputFunctionsParser(
            pydantic_schema=TopicClassifier, attr_name="topic"
        )
        classifier_chain = llm | parser

        chain = (RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain)
                 | prompt_branch
                 )
        # second_chain = prompt_branch
        result_prompt = chain.invoke({"input": user_query})
        return result_prompt.text

    def _get_response_from_base_gpt(self, prompt_in: str, role: Role = Role.PLANNER):
        print(f"{BLUE}STARTING LANGCHAIN.LLM \
              {RESET}: {YELLOW}{str(role)}{RESET}")
        messages = [HumanMessage(content=prompt_in)]
        res = self.model.invoke(messages).content
        print(f"{BLUE}LANGCHAIN.LLM MESSAGE{RESET}: {res}")
        return res

    def _get_response_from_assistant(self, prompt_in: str, role: Role = Role.PLANNER):
        print(f"{BLUE}STARTING OPENAI ASSISTANT\
              {RESET}: {YELLOW}{str(role)}{RESET}")

        # init_instructions = "You are an AI data analyst and your job is to assist the user with simple data analysis."
        # if role == Role.CODER:
        #     init_instructions += " You generate clear and simple code following the given instructions."

        thread = self.client.beta.threads.create()

        # Add a message to a thread
        # message = client.beta.threads.messages.create(
        #     thread_id=thread.id,
        #     role="user",
        #     content="",
        # )

        # Run the assistant (start it)
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.my_assistant.id,
            instructions=prompt_in,
        )

        # messages = client.beta.threads.messages.list(thread_id=thread.id)
        # print(f"{BLUE}USER MESSAGE{RESET}: {messages.data[-1].content[0].text.value}")

        # Get the status of a Run
        run = self.client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id)
        count = 0
        while run.status != "completed" and run.status != "failed":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id)
            print(f"{YELLOW}    RUN STATUS{RESET}: {run.status}")
            time.sleep(1.5)
            count += 1

        if run.status == "failed":
            return None

        # print(f"{BLUE}RUN OBJECT{RESET}: {run}")
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        print(f"{BLUE}ASSISTANT MESSAGE{RESET}: \
              {messages.data[0].content[0].text.value}")
        return messages.data[0].content[0].text.value

    def plan_steps_with_gpt(self, user_query, df, save_plot_name, query_type=None, data_annotation: dict | None = None):
        assert query_type is not None, "query_type must be specified (tagged before calling this function)"

        temlate_for_plot_planner = self.prompts.generate_steps_for_plot_show_prompt(
            df, user_query, data_annotation)
        if save_plot_name is not None:
            temlate_for_plot_planner = self.prompts.generate_steps_for_plot_save_prompt(
                df, user_query, save_plot_name, data_annotation)
        temlate_for_math_planner = self.prompts.generate_steps_no_plot_prompt(
            df, user_query, data_annotation)

        selected_prompt = temlate_for_math_planner
        if query_type == "plot":
            selected_prompt = temlate_for_plot_planner

        return self._call_openai_llm(selected_prompt, role=Role.PLANNER), selected_prompt

    def query_rewrite(self, user_query, df, save_plot_name=None, query_type=None, data_annotation: dict | None = None):

        template_prompt = self.prompts.query_rewrite(
            df, user_query, data_annotation)

        return self._call_openai_llm(template_prompt, role=Role.PLANNER), template_prompt

    def generate_replan(self, user_query, df, plan, save_plot_name=None, query_type=None, data_annotation: dict | None = None):
        template_prompt = self.prompts.generate_replan(
            df, user_query, plan, data_annotation)
        return self._call_openai_llm(template_prompt, role=Role.PLANNER)

    def merge_query_and_history(self, user_query, history):
        template_prompt = self.prompts.get_merge_query_and_history(
            user_query, history)
        return self._call_openai_llm(template_prompt, role=Role.PLANNER)

    def is_query_clear(self, user_query, df, data_annotation: dict | None = None):
        template_prompt = self.prompts.is_query_clear(
            df, user_query, data_annotation)
        return self._call_openai_llm(template_prompt, role=Role.PLANNER)

    def is_coding_needed_cls(self, user_query, df) -> int:
        # 0 - no coding required, 1 - coding required
        template_prompt = self.prompts.is_coding_needed_cls(df, user_query)
        answer = self._call_openai_llm(template_prompt, role=Role.PLANNER)
        print(answer)
        if '1' in answer:
            return 1
        return 0

    def answer_noncoding_query(self, user_query, chat_history):
        template_prompt = self.prompts.get_answer_noncoding_query_prompt(
            user_query, chat_history)
        return self._call_openai_llm(template_prompt, role=Role.PLANNER)

    def modify_query_for_save_df(self, user_query):
        return self.prompts.get_save_df_prompt(user_query)

    def query_disambiguation(self, user_query, df, data_annotation: dict | None = None):
        template_prompt = self.prompts.query_disambiguation(
            df, user_query, data_annotation)
        return self._call_openai_llm(template_prompt, role=Role.PLANNER)

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
            return GPTCoder().query(self.model, instruction_prompt), instruction_prompt
        elif llm == "codellama/CodeLlama-7b-Instruct-hf":  # local llm
            answer, self.local_coder_model = CodeLlamaInstructCoder().query(llm,
                                                                            instruction_prompt,
                                                                            already_loaded_model=self.local_coder_model,
                                                                            adapter_path=adapter_path)
            return answer, instruction_prompt

        elif llm.startswith("WizardLM/WizardCoder-"):  # under 34B
            return WizardCoder().query(llm, instruction_prompt), instruction_prompt

        elif llm == "ise-uiuc/Magicoder-S-CL-7B":  # doesn't really work yet
            return CodeLlamaInstructCoder().query(llm, instruction_prompt), instruction_prompt

    def fix_generated_code(self, df, code_to_be_fixed, error_message, user_query, initial_coder_prompt, data_annotation: dict | None = None):
        prompt = self.prompts.fix_code_prompt(
            df, user_query, code_to_be_fixed, error_message, initial_coder_prompt)
        return self._call_openai_llm(prompt, Role.DEBUGGER), prompt

    def pure_openai_assistant_answer(self, df_filename, user_query, possible_plotname=None):
        def wait_on_run(run_local, thread_local):
            while run_local.status == "queued" or run_local.status == "in_progress":
                run_local = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_local.id,
                    run_id=run_local.id,
                )
                time.sleep(0.5)
            return run_local

        def pretty_print(messages_local):
            ret_val = ""
            for m in messages_local:
                # ret_val += f"{m.role}: {m.content[0].text.value}\n"

                # and has attribute text
                if len(m.content) != 0 and hasattr(m.content[0], "text"):
                    ret_val += f"{m.content[0].text.value}\n"
            return ret_val

        def get_response(thread):
            return self.client.beta.threads.messages.list(thread_id=thread.id)

        def get_file_ids_from_thread(thread):
            file_ids_local = [
                file_id
                for m in get_response(thread)
                for file_id in m.file_ids
            ]
            return file_ids_local

        def write_file_to_temp_dir(file_id, output_path):
            file_data = self.client.files.content(file_id)
            file_data_bytes = file_data.read()
            with open(output_path, "wb") as f:
                f.write(file_data_bytes)

        file = self.client.files.create(
            file=open(
                df_filename,
                "rb",
            ),
            purpose="assistants",
        )

        self.my_assistant = self.client.beta.assistants.update(
            self.my_assistant.id,
            tools=[{"type": "code_interpreter"}],
            file_ids=[file.id],
        )

        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=user_query
        )

        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.my_assistant.id,
        )

        wait_on_run(run, self.thread)

        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id,
            order="asc",
            after=message.id
        )

        # print(run.file_ids) # still only dataframe file id

        file_ids = get_file_ids_from_thread(self.thread)
        print(f"{BLUE}RETURNED FILE IDS{RESET}: {file_ids}")
        if len(file_ids) == 0:
            return pretty_print(messages)
        some_file_id = file_ids[0]
        try:
            write_file_to_temp_dir(some_file_id, possible_plotname)
        except Exception as e:
            print(f"{RED}ERROR WRITING FILE{RESET}: {e}")
            return pretty_print(messages)
        return pretty_print(messages) + " Plot saved to " + possible_plotname
    
    def generate_title(self, prompt: str) -> str:
        try:
            response = self._call_openai_llm(prompt, role=Role.PLANNER)
            title = response.strip() 
            return title
        except Exception as e:
            print(f"Error generating title: {e}")
            return "Untitled Conversation"