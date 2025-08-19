import os
import json
from typing import Callable
from random import getrandbits
import pandas as pd
from pathlib import Path
import matplotlib
# AG: patch for 'RuntimeError: main thread is not in main loop';
# In Matplotlib, the Agg backend is a non-interactive backend that renders figures to PNG files.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Optional
from .llms import LLM
from .code_manipulation import Code
from .logger import *
import threading
from .preprocessing import *
from functools import partial
from time import time
from .AgentDataFrameManager import AgentDataFrameManager, AddDataMode
from copy import copy
from .data_classes import CodeSnippet
import atexit
import shutil


def cleanup(folder_path):
    # Check if the folder exists before attempting to remove it
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)  # Clean up directory and all its contents
            print(f"Cleaned up folder: {folder_path}")
        except Exception as e:
            print(f"Error while cleaning up: {e}")


class TableAgent:
    def __init__(
        self,
        gpt_model: Callable,
        table_file_path: str | list[str] | None = None,
        max_debug_times: int = 2,
        coder_model: str = 'gpt',
        adapter_path: str = '',
        head_number: int = 2,
        prompt_strategy: str = 'simple',
        prompt_name: str = 'generate_whatever',
        # [remove_spaces_in_column_names],
        preprocessing_steps: list[Callable[[pd.DataFrame], pd.DataFrame]] = [],
        functions_description: str = None,
        functions_list: list[Callable] = None,
        generated_code_exec_timeout: int | None = 300,
        data_specs_dir_path: str | None = None,
        tagging_strategy="openai",
        query_type=None,
        is_query_ok_llm=None,
        allow_same_process_execution: bool = False,
        tmp_file_path=None,
        column_description_paths=None,
    ):
        """
        Creates AgentTBN

        Parameters
        ----------
        table_file_path: str
            Path to the data which will be used

        gpt_model: Callable
            Which LLM model to use

        max_debug_times: int = 2
            How many times at most the agent can try to repair its generated code.

        coder_model: str, default='gpt'
            Name of the coder model

        adapter_path: str, default=''
            TODO

        head_number: int, default=2
            The amount of intial rows given in the prompts to the agent
            as the example of the dataframe being used.        

        prompt_strategy: str, default='simple'
            Determines which prompt strategy to use. Is one of following:
            {'functions', 'simple', 'coder_only_simple', 'coder_only_functions'}

        preprocessing_steps: list[Callable[[pd.DataFrame], pd.DataFrame]], default=[]
            List of functions accepting, altering and at the end returning pd.DataFrame which
            the self._df is always ran through when it is accessed.

        functions_description: str = None
            Description of the 'pre-cooked' functions which are available for the agent to use
            being supplied in the `functions_list` constructor parameter. 

        functions_list: list[Callable], default=None
            The list of 'pre-cooked' functions which are available for the agent to use.
            If left as default None it is interpreted as empty list.

        generated_code_exec_timeout: int | None, default=300
            The time in seconds for how long the generated code is left to be executed
            before it gets forcefully terminated. If set as None it will be always 
            waited for the generated code to completely execute (possibly never)

        data_specs_dir_path: str | None = None
            Path to the directory containing JSON specifications of all of the data.
            If kept as default None then no data specifications informations are added to the prompts.

            The used JSON specification is the one which shares the name with the used dataframe (If not found then None is used)        

        query_type: Optional[str], default=None
            Type of the query.

        allow_same_process_execution: bool, default=False
            In case of `generated_code_exec_timeout` being None this toggles whether the agent's
            generated code is being ran in the same process (faster) or separate process (safer)
        """

        self.gpt_model = gpt_model
        self.coder_model = coder_model
        self.adapter_path = adapter_path
        self.max_debug_times = max_debug_times
        self.preprocessing_steps = preprocessing_steps
        # self.prompt_strategy = prompt_strategy
        self.prompt_name = prompt_name
        self.head_number = head_number
        self.generated_code_exec_timeout = generated_code_exec_timeout
        self.data_specs_dir_path = data_specs_dir_path
        self.allow_same_process_execution = allow_same_process_execution
        self.agent_hash = '%032x' % getrandbits(128)
        self.tmp_file_path = os.path.join(tmp_file_path, f'df_saves_{self.agent_hash}') if tmp_file_path else os.path.join(__file__, 'temp', f'df_saves_{self.agent_hash}')

        # To skip the reasoning part for one run:
        self._plan = None
        self._tagged_query_type = None
        self._prompt_user_for_planner = None

        # Attributes to be infered only once at initialization
        self.provider = "openai" if self.coder_model == "gpt" else "local"
        self.functions_list = [] if functions_list is None else functions_list

        self.use_assistants_api = False
        assert not (self.use_assistants_api and prompt_strategy ==
                    "coder_only_simple"), "Both use_assistants_api and coder_only_simple cannot be True at the same time."

        self.llm_calls = LLM(
            use_assistants_api=self.use_assistants_api,
            model=self.gpt_model,
            head_number=self.head_number,
            prompt_strategy=prompt_strategy,
            functions_description=functions_description
        )

        self.agent_dataframe_manager = AgentDataFrameManager(
            table_file_paths=table_file_path,
            data_specs_dir_path=data_specs_dir_path,
            llm_calls=self.llm_calls,
            forced_data_specs=column_description_paths
        )


        self._code = Code(
            self.agent_dataframe_manager.get_dataframes(),
            self.agent_dataframe_manager.get_dataframes_source_filenames(),
            self.functions_list,
            self.tmp_file_path,
        )
        atexit.register(cleanup, self._code._df_saving_dir_path)

        # So the `self._code` gets notified when the `self.agent_dataframe_manager` changes its stored dataframes
        self.agent_dataframe_manager.attach(self._code, 'df_change')

        self.prompt_strategy = self.llm_calls.prompts.strategy
        if self.tmp_file_path:
            self.prompt_strategy.tmp_file_path = self.tmp_file_path

        if is_query_ok_llm:
            self.is_query_ok_llm = LLM(
                use_assistants_api=self.use_assistants_api,
                model=is_query_ok_llm,
                head_number=self.head_number,
                prompt_strategy='simple',
                functions_description=functions_description
            )
        else:
            self.is_query_ok_llm = None

        # So that df.head(1) did not truncate the printed table
        pd.set_option('display.max_columns', None)
        # So that did not insert new lines while printing the df
        pd.set_option('display.expand_frame_repr', False)

    @property
    def table_file_path(self):
        """
        Path to the dataframe.
        """
        return self.agent_dataframe_manager.get_table_file_paths()

    @table_file_path.setter
    def table_file_path(self, table_file_path: str | list[str], forced_data_specs: list[dict | None] | dict | None = None):
        """
        Reloads dataframes which the agent uses.
        """
        self.agent_dataframe_manager.remove_all_data()
        self.agent_dataframe_manager.add_data(
            table_file_path, AddDataMode.APPEND, forced_annotations=forced_data_specs
            ) 

    @property
    def df(self):
        """
        Datframe(s) the agent operates on.
        """
        return self.agent_dataframe_manager.get_dataframes(return_copy=True)

    @df.setter
    def df(self, _):
        raise Exception(
            'df is read only and set via `self.agent_dataframe_manager`.')

    @property
    def filename(self):
        return self.agent_dataframe_manager.get_dataframes_source_filenames()

    @filename.setter
    def filename(self, _):
        raise Exception('filename is read only.')

    @property
    def data_specs(self):
        """
        Loaded JSON file containing data specifications for the data.
        If data_specs_dir_path is specified and contains a matching file, it loads the specifications from there.
        Otherwise, it uses the generated specifications.
        """
        return self.agent_dataframe_manager.get_data_specs()

    @data_specs.setter
    def data_specs(self, _):
        raise Exception('data_specs is read only.')

    def save_data_specs(self, data_specs, table_name):
        """
        Saves the given data specifications to a JSON file.
        """
        data_specs_dir = os.path.abspath(os.path.join(
            __file__, '..', '..', '..', 'data', 'specifications'))

        # Ensure the directory exists
        os.makedirs(data_specs_dir, exist_ok=True)
        data_specs_path = os.path.join(data_specs_dir, f"{table_name}.json")
        try:
            with open(data_specs_path, 'w') as f:
                json.dump(data_specs, f, indent=2)
            print(f"Data specs saved to {data_specs_path}")
        except Exception as e:
            print(f"Failed to save data specs: {e}")

    def load_new_table(self, table_file_path, regenerate_spec=True):
        """
        Loads a new dataframe into the agent and regenerates data specifications if necessary.
        """
        if regenerate_spec:
            self.agent_dataframe_manager.add_data(
                table_file_path, add_data_mode=AddDataMode.CHECK_RELOAD)
        else:
            self.agent_dataframe_manager.add_data(
                table_file_path, add_data_mode=AddDataMode.CHECK_APPEND)

    def skip_reasoning_part(self, plan: str, tagged_query_type: str, prompt_user_for_planner: str):
        self._plan = plan
        self._tagged_query_type = tagged_query_type
        self._prompt_user_for_planner = prompt_user_for_planner
        if isinstance(self._plan, str) != isinstance(self._tagged_query_type, str):
            raise Exception(
                "Both plan and tagged_query_type must be either None or a string.")

    def _reset_skip_reasoning_part(self):
        self._plan = None
        self._tagged_query_type = None
        self._prompt_user_for_planner = None

    def _test_if_query_is_ok(self, query: str, llm=None):
        '''
        Expects GPT4o
        '''
        if llm is None:
            llm = self.llm_calls

        is_query_ok = llm.is_query_clear(
            query, self.df, data_annotation=self.data_specs)
        if "Yes" in is_query_ok:
            return True, is_query_ok.replace("Yes", "").replace("No", "").replace(", | clarifying_query:", "Note: I did not fully understand the question.").strip()

        if len(is_query_ok.split("_")) < 3 and 'do you mean' in is_query_ok.lower():
            return True, is_query_ok.replace("Yes", "").replace("No", "").replace(", | clarifying_query:", "Note: I did not fully understand the question.").strip()

        return False, is_query_ok.replace("Yes", "").replace("No", "").replace(", | clarifying_query:", "Note: I did not fully understand the question.").strip()

    def _test_query_disambiguation(self, query: str):
        is_query_ok = self.llm_calls.query_disambiguation(
            query, self.df, data_annotation=self.data_specs)
        return is_query_ok, ""
    
    def get_query_sufix_with_og_df_names(self) -> str:
        """
        Returns sufix for the query which ties df_<i> (or df) with the original 
        source filenames of their.
        """
        loaded_source_filenames: list[str] | str = self.agent_dataframe_manager.get_dataframes_source_filenames()

        if type(loaded_source_filenames) is list:
            df_vars = [f'df_{i + 1}' for i in range(len(loaded_source_filenames))]
            df_og_names = loaded_source_filenames
        else:
            df_vars = ['df']
            df_og_names = [loaded_source_filenames]

        return '\nNote that: ' + ', '.join([
            f'original (source file) name for dataframe {df_var} is {df_og_name}' for (df_var, df_og_name) in zip(df_vars, df_og_names)
        ]) + '.'

    def answer_query(
        self,
        query: str,
        show_plot=False,
        save_plot_dir="temp/",
        do_query_rewrite=False,
        check_plan=False,
        history: Optional[List[Tuple]] = None,
        check_query_ok=False,
        tested_type=None,
        save_df=False
    ) -> Tuple[str, List[CodeSnippet]] | Tuple[str, List[CodeSnippet], str, str]:
        """
        for history format, see transform_history function in GUI/main.py

        Additionally returns a dictionary with info:
            - Which prompts were used and where,
            - Generated code,
            - Number of error corrections etc.
        """
        if len(self.functions_list) != 0:
            query += ' When you need help with any step, use predefined function you are provided.'

        # Add names of the tables of dataframe variables to the query
        query += self.get_query_sufix_with_og_df_names()

        possible_plotname = None
        if not show_plot:  # No need to plt.show()
            # this name is only passed as argument to prompt_template. In the generated code, the filename
            # will be replaced by this string: <code_segment_is>_SEG_plot_<agent_hash>.png
            possible_plotname = self.prompt_strategy.create_possible_plot_name(
                self.agent_hash)

        if history:
            query = self.llm_calls.merge_query_and_history(query, history)

        if do_query_rewrite:
            query, _ = self.llm_calls.query_rewrite(
                query, self.df, data_annotation=self.data_specs)

        # We always want the answer to be in the df variable, as the same
        # code can be used for saving the dataframe
        if True:
            query = self.llm_calls.modify_query_for_save_df(query)

        if check_query_ok and self.is_query_ok_llm:

            def run_test_if_query_is_ok(query, ret):
                answer, comment = self._test_if_query_is_ok(
                    query, llm=self.is_query_ok_llm)
                ret['answer'] = answer
                ret['comment'] = comment

            ret = {}
            thread = threading.Thread(
                target=run_test_if_query_is_ok, args=(query, ret))
            thread.start()
        else:
            thread, ret = None, None

        self._tagged_query_type = self.llm_calls.tag_query_type(query)
        if self._plan is None and self.prompt_strategy.has_planner:  # Not skipping the reasoning part
            self._plan, self._prompt_user_for_planner = self.llm_calls.plan_steps_with_gpt(
                query, self.df, save_plot_name=possible_plotname, query_type=self._tagged_query_type, data_annotation=self.data_specs)
            if check_plan:
                self._plan = self.llm_calls.generate_replan(
                    query, self.df, self._plan, data_annotation=self.data_specs)

        generated_answer, coder_prompt = self.llm_calls.generate_code(
            query,
            self.df,
            self._plan,
            show_plot=show_plot,
            tagged_query_type=self._tagged_query_type,
            llm=self.coder_model,
            adapter_path=self.adapter_path,
            save_plot_name=possible_plotname,  # for the "coder_only" prompt strategies
            data_annotation=self.data_specs,
            prompt_name=self.prompt_name
        )

        print(f"Generated code: {generated_answer}")
        print(self._tagged_query_type)

        # 'local' removes the definition of a new df if there is one
        code_to_execute = Code.extract_code(
            generated_answer, provider=self.provider, show_plot=show_plot, prompt_strategy=self.prompt_strategy)

        result_list = self.prompt_strategy.run_code_segments(
            copy(code_to_execute),
            self._code,
            query,
            coder_prompt,
            self.agent_hash,
            tested_type,
            args={
                'tagged_query_type': self._tagged_query_type,
                'timeout': self.generated_code_exec_timeout,
                'save_df': save_df,
                'same_proc_exec': self.allow_same_process_execution
            },
            n_dfs=1 if not isinstance(self.df, list) else len(self.df),
            fix_code_func=self.fix_code,
            save_plot_dir=save_plot_dir
        )

        res = self.prompt_strategy.formulate_result(
            result_list, generated_answer)
        ret_value = res
        if res == "":
            if self._tagged_query_type == "general":
                ret_value = "Empty output from the exec() function for the text-intended answer."
            elif self._tagged_query_type == "plot":
                ret_value = result_list[0]['plot_filenames']

        self._reset_skip_reasoning_part()

        if check_query_ok and thread:
            thread.join()
            return ret_value, result_list, ret['answer'], ret['comment']

        return ret_value, result_list
    
    def generate_conversation_title(self, prompts: List[str], df_name: str = None, column_names: List[str] = None) -> str:   
        # Combine the first three user messages into a single prompt for name generation
        prompts_collected = " ".join([f"Prompt {i+1}: {msg}" for i, msg in enumerate(prompts)])
        
        if df_name and column_names:
            df_info = f"\nDataframe: {df_name}\nColumns: {', '.join(column_names)}"
        else:
            df_info = ""

        # Create the final query for the LLM, including dataframe and columns
        prompt_for_name_generation = f"""
        Based on these prompts: {prompts_collected}{df_info}, generate a brief and high-level title for the conversation.
        The title should summarize the theme without specific details or code.
        You can use words like 'Analyzing', 'Exploring', or 'Understanding' for variety, but avoid using 'Analyzing' every time.
        Avoid including the phrases 'Prompt' or 'Based on' in the title.
        """

        try:
            title = self.llm_calls.generate_title(prompt_for_name_generation)

            # Clean and format the title
            cleaned_title = self.clean_conversation_name(title)
            return cleaned_title

        except Exception as e:
            print(f"Error while generating conversation title: {e}")
            return "General Data Exploration"
        
    def clean_conversation_name(self, name: str) -> str:
        # Strip unwanted symbols like **, #, and spaces
        name = name.replace('*', '').replace('#', '').strip()

        # Capitalize the first letter of the cleaned name
        return name.capitalize()

    def fix_code(self, code_to_execute, traceback, query, coder_prompt, n_dfs=1):
        regenerated_code, debug_prompt = self.llm_calls.fix_generated_code(
            self.df, code_to_execute, traceback, query, coder_prompt, self.data_specs)
        code_to_execute = Code.extract_code(
            regenerated_code, provider=self.provider, prompt_strategy=self.prompt_strategy)[0]
        code_to_execute = Code.preprocess_extracted_code(
            code_to_execute, self.prompt_strategy, n_dfs=n_dfs)
        return code_to_execute, debug_prompt
