from time import time
from os import path
import re
import os
import random
from copy import copy
from ..data_classes import CodeSnippet


class BasePromptStrategy:

    base_prompt = """You are highly proficient with Python and the pandas library. A user has submitted a query that you need to address: '{input}'. 
You also have a list of subtasks that need to be completed. However, no data are available yet so you cannot make any analysis.
Thus, for now, till the data are not available, do not use any coding and just react textually to the user input. You must not generate code! Your output
should be helpful and natural and nice, you do not have to bother with potential results and where to save them.

Your answer:"""

    max_debug_times = None

    def format(self, prompt_name, df, user_query, plan, column_description, functions_description, **kwargs):
        pass

    def append_to_code(self, extracted_code, n_dfs=None):
        return extracted_code

    def formulate_result(self, code_segments_obj, full_answer):
        # assumes just one code snipet and hence only one result by default
        if code_segments_obj is None:
            return full_answer
        return code_segments_obj[0].result_of_execution

    def map(self, code_segments, func, args):
        code_res = '\n'.join(code_segments).lstrip()
        return [func(code_res, **args)]

    def run_code_segments(self,
                          code_segments,
                          code_class,
                          query,
                          coder_prompt,
                          agent_hash,
                          tested_type=None,
                          tested_type_args: dict = None,
                          args: dict = {},
                          n_dfs=1,
                          fix_code_func=None,
                          segment_index=-1,
                          save_plot_dir=""
                          )-> list[CodeSnippet] | None:
        if len(code_segments) == 0 or n_dfs == 0:
            return None
     
        if segment_index != -1:
            return [
                self.execute_code(
                    code_segments[segment_index],
                    code_class,
                    query,
                    coder_prompt,
                    agent_hash,
                    tested_type=tested_type,
                    tested_type_args=tested_type_args,
                    args=args,
                    n_dfs=n_dfs,
                    fix_code_func=fix_code_func,
                    possible_plotname_prefix=save_plot_dir
                )
            ]

        return [
            self.execute_code(
                code_to_execute,
                code_class,
                query,
                coder_prompt,
                agent_hash,
                tested_type=tested_type,
                tested_type_args=tested_type_args,
                args=args,
                n_dfs=n_dfs,
                fix_code_func=fix_code_func,
                possible_plotname_prefix=save_plot_dir + str(idx) + "_SEG_plot_") for idx, code_to_execute in enumerate(code_segments)
        ]

    def execute_code(self,
                     code_to_execute,
                     code_class,
                     query,
                     coder_prompt,
                     agent_hash,
                     tested_type=None,
                     tested_type_args: dict = None,
                     args: dict = {},
                     n_dfs=1,
                     fix_code_func=None,
                     possible_plotname_prefix=""
                     ):

        code_snipet_obj = CodeSnippet()

        code_to_execute_backup = code_to_execute
        t_start_query_answering = time()

        code_snipet_obj.code_segment = copy(code_to_execute)

        try:
            code_to_execute = code_class.preprocess_extracted_code(
                code_to_execute, self, n_dfs=n_dfs)
            if tested_type:
                code_to_execute = code_to_execute + \
                    tested_type.to_saving_code_snippet(
                        self.tmp_file_path, var_name='result')
        except Exception as e:
            print(e)

        if args['tagged_query_type'] == "plot":
            n_plots = self.count_savefig_calls(code_to_execute)
            plot_names = [possible_plotname_prefix +
                          self.create_possible_plot_name(agent_hash) for i in range(n_plots)]
            code_to_execute = self.rename_plots(code_to_execute, plot_names)
        else:
            plot_names = []

        print("code execution:")
        print(code_to_execute)

        res, traceback, exec_exit_reason, t_elapsed, path_to_saved_dfs = code_class.execute_generated_code(
            code_to_execute,
            **args
        )

        debug_prompt = ""

        count = 0
        errors = []

        if traceback == "empty exec()":
            res = "ERROR"
            errors.append(traceback)

        while res == "ERROR" and count < self.max_debug_times and fix_code_func is not None:

            errors.append(traceback)
            code_to_execute, debug_prompt = fix_code_func(
                code_to_execute, traceback, query, coder_prompt, n_dfs=n_dfs)

            res, traceback, exec_exit_reason, t_elapsed, path_to_saved_dfs = code_class.execute_generated_code(
                code_to_execute,
                **args
            )
            count += 1
        errors = errors + \
            [traceback] if res == "ERROR" or not code_to_execute.strip() else []

        code_snipet_obj.query_type = args['tagged_query_type']
        code_snipet_obj.count_of_fixing_errors = count
        code_snipet_obj.final_code_segment = code_to_execute
        code_snipet_obj.last_debug_prompt = debug_prompt
        code_snipet_obj.successful_code_execution = res != "ERROR"
        code_snipet_obj.result_of_execution = res
        code_snipet_obj.plot_filenames = plot_names
        code_snipet_obj.path_to_saved_dfs = path_to_saved_dfs
        code_snipet_obj.code_errors = '\n'.join([f"{index}. \"{item}\"" for index, item in enumerate(errors)])
        code_snipet_obj.traceback = traceback
        code_snipet_obj.wall_time_final_code_runtime = t_elapsed
        code_snipet_obj.wall_time_full = time() - t_start_query_answering
        code_snipet_obj.exec_exit_reason = exec_exit_reason

        # ret_value = res

        return code_snipet_obj #ret_value, details, exec_exit_reason

    def rename_plots(self, code, names):
        return code

    def count_savefig_calls(self, code_line):
        """
        This function counts the number of savefig calls in a given code line.
        It can handle both plt.savefig and <plot_var>.savefig cases.

        Parameters:
        code_line (str): The line of code containing the savefig pattern.

        Returns:
        int: The number of savefig calls in the code line.
        """
        # Regular expression pattern to match plt.savefig('{save_plot_name}') or <plot_var>.savefig('{save_plot_name}')
        pattern = r"\b\w+?\.savefig\('.*?'\)"

        # Find all matches
        matches = re.findall(pattern, code_line)

        # Return the number of matches
        return len(matches)

    def create_possible_plot_name(self, id):
        possible_plotname = os.path.splitext(os.path.basename(id))[0] + str(
            random.randint(10, 999)) + ".png"
        return possible_plotname
    
    def format(self, prompt_name, df, user_query, plan, column_description, functions_description, **kwargs):
        if isinstance(df, list) and len(df) == 0:
            return self.base_prompt.format(input=user_query)
        
        if prompt_name == 'generate_steps_no_plot':
            if hasattr(self, 'generate_steps_no_plot'):
                return self.format_generate_steps_no_plot_prompt(
                    head_number=kwargs['head_number'],
                    df=df,
                    user_query=user_query,
                    column_description=column_description,
                    functions_description=functions_description
                )
            else:
                raise Exception(
                    "This prompt strategy does not support generating steps., you should directly call a prompt-specific format method."
                )
        elif prompt_name == 'generate_code':
            prompt = self.generate_code if kwargs['use_gpt4o'] is False else self.generate_code_gpt4_
            return prompt.format(
                input=user_query,
                df_head=df.head(kwargs['head_number']),
                plan=plan,
                head_number=kwargs['head_number'],
                column_description=column_description,
                functions_description=functions_description
            )

        elif prompt_name == 'generate_code_multiple_dfs':
            data = "\n"
            for i in range(len(df)):
                data += f"DataFrame df_{i+1}:\n{df[i].head(kwargs['head_number'])}\n\
                {column_description[i]}\n"

            return self.generate_code_multiple_dfs.format(
                input=user_query,
                plan=plan,
                functions_description=functions_description,
                data=data,
                df_examples=", ".join([f"df_{i+1}" for i in range(len(df))])
            )

        elif prompt_name == 'generate_whatever':
            return self.generate_whatever.format(
                input=user_query,
                df_head=df.head(kwargs['head_number']),
                head_number=kwargs['head_number'],
                column_description=column_description,
            )

        else:
            raise Exception("Unsupported prompt strategy")


class FunctionBasePromptStrategy(BasePromptStrategy):

    def append_to_code(self, extracted_code, n_dfs=None):
        if n_dfs is None or n_dfs <= 1:
            return extracted_code + "\n\n" + "result = solve(df)\nprint(result)"
        else:
            return extracted_code + "\n\n" + f"result = solve({', '.join([f'df_{i+1}' for i in range(n_dfs)])})\nprint(result)"
