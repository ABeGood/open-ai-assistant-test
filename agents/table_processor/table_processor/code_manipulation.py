from .logger import *
from .AgentDataFrameManager import AgentDataFrameManager
from .Observer import Subscriber
from .prompts_dir.base_prompt_strategy import BasePromptStrategy
import io
import re
import os
import sys
import copy
import shutil
import autopep8
import traceback
from time import time, sleep
from enum import Enum
from typing import Callable, Optional, List
from contextlib import redirect_stdout
from multiprocessing import Process, Queue

import ast
import pandas as pd
import threading
from time import time, sleep
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, "prompts"))


class ExecExitReason(Enum):
    """
    Used for indicating how the generated code executed
    """
    SUCCESS = 0
    EXCEPTION = 1
    TIMEOUT = 2


def extract_solve_function_params(code_str: str) -> List[str]:
    """
    Parse the code to extract parameters from the solve function.
    """
    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.params = []

        def visit_FunctionDef(self, node):
            if node.name == 'solve':
                self.params = [arg.arg for arg in node.args.args]
            # No need to continue after finding solve function
            return

    tree = ast.parse(code_str)
    visitor = FunctionVisitor()
    visitor.visit(tree)
    return visitor.params

def modify_code_str(code_str: str) -> str:
    """
    Modify the code_str to assign result of solve to original dataframes
    only if the result contains DataFrames.
    """
    params = extract_solve_function_params(code_str)
    
    # Create conditional assignment string based on the number of params
    if len(params) == 0:
        return code_str
    
    assignment_str = ""
    if len(params) == 1:
        assignment_str = (
            f"if isinstance(result, pd.DataFrame):\n"
            f"    {params[0]} = result\n"
        )
    else:
        # Build tuple unpacking logic with careful checks
        checks_and_assignments = "\n".join([
            f"if isinstance(result, tuple) and len(result) == {len(params)}:",
            *[
                f"    if isinstance(result[{i}], pd.DataFrame): {params[i]} = result[{i}]"
                for i in range(len(params))
            ]
        ])
        
        assignment_str = checks_and_assignments + "\n"

    # Look for solve function call in the code and append the assignment
    solve_call_str = "result = solve("
    if solve_call_str in code_str:
        insertion_index = code_str.rfind(solve_call_str) + len(solve_call_str)
        code_str = code_str[:insertion_index] + code_str[insertion_index:] + assignment_str

    return code_str


def run_code_execution_process(exec_kwargs: dict, df_kwargs_keys: dict[str, str], mp_input_queue: Queue, mp_output_queue: Queue) -> None:
    """
    Target function for code execution process. Communication occurs via given queues.
    """
    def deepcopy_kwargs(exec_kwargs: dict):
        return {
            k: copy.deepcopy(v) for (k, v) in exec_kwargs.items()
        }

    # Infinite loop
    while True:

        # Get the source code to execute
        code_str, dir_path, save_result_only = mp_input_queue.get()

        with redirect_stdout(io.StringIO()) as output:
            # Try to run the code and let its std::out be noted in `output`
            try:
                # We deepcopy exec_kwargs so the df stays constant thoughout the executions
                tmp_exec_kwargs = deepcopy_kwargs(exec_kwargs)

                # The alteration of code here is not propagated to the rest of the code
                # Additionaly it breaks dataframe saving functionality of GUI
                # if dir_path is not None:
                #     code_str = modify_code_str(code_str)                
                exec(code_str, tmp_exec_kwargs)

                if dir_path is not None:
                    save_dfs_from_exec_kwargs(tmp_exec_kwargs, df_kwargs_keys, dir_path, save_result_only)

            # In case code execution dies we return the exception as string
            except Exception:

                exc_type, exc_value, _ = sys.exc_info()
                full_traceback = traceback.format_exc()
                # Filter the traceback
                exec_traceback = Code.filter_exec_traceback(
                    full_traceback, exc_type.__name__, str(exc_value))

                mp_output_queue.put((
                    ExecExitReason.EXCEPTION,
                    exec_traceback,
                    dir_path
                ))

                # Skip the rest of the loop and wait for new source code to execute
                continue

            # If no Exception was caught then return the program's output
            mp_output_queue.put((
                ExecExitReason.SUCCESS,
                output.getvalue(),
                dir_path
            ))


def save_dfs_from_exec_kwargs(exec_kwargs: dict, df_kwargs_keys: dict[str, str], dir_path: str, save_result_only: bool):
    """
    Saves the DataFrames within `exec_kwargs` into directory given by `dir_path`
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    os.makedirs(dir_path)

    for df_exec_name, df_og_name in df_kwargs_keys.items():
        
        if save_result_only and df_exec_name != 'result' and df_og_name != 'result':
            continue

        # Since `result` may remain None the to_csv call would fail
        try:
            exec_kwargs[df_exec_name].to_csv(
                os.path.join(dir_path, f'{df_og_name}.csv')
            )
        except:
            pass


class Code(Subscriber):
    """
    Class for code execution.

    Implements topics:
        - df_change
    """

    def __init__(
        self,
        df: pd.DataFrame | list[pd.DataFrame],
        df_original_name: str | list[str] | None,
        functions_list: list[Callable],
        df_saving_dir_path: str = os.path.join(__file__, '..', 'df_saves')
    ) -> None:
        """
        Creates Code instance which is used for executing code generated by agent

        Parameters
        ----------
        df: pd.DataFrame | list[pd.DataFrame]
            dataframe(s) which will be used as context for code execution.
            Can be changed by calling `self.update_df`

        df_original_name: str | list[str] | None
            Orignal name(s) of the dataframe(s) used for potential saving of the dataframe(s).
            In case of `None`, the df_<num> naming notation is being used.

        functions_list: list[Callable]
            list of functions which will be used as context for code execution.
            Can be changed by calling `self.update_functions_list`

        df_saving_dir_path: str, default=os.path.join(__file__, '..', 'df_saves')
            Path to the directory where the potentionaly saved dfs will be.
        """
        super().__init__()
        self._df = df
        self._df_original_name = df_original_name

        self._functions_list = functions_list
        self._df_saving_dir_path = df_saving_dir_path

        self._proc: Process = None
        self._start_new_code_executing_process()

    def update(self, publisher: AgentDataFrameManager, topic: str):
        """
        Updates the Code instance according to `topic`
        """
        if topic == 'df_change':
            self.update_df(publisher.get_dataframes(),
                           publisher.get_dataframes_source_filenames())

    def __del__(self):
        """
        Kill your child process in destructor
        """
        self._kill_code_executing_process()

    def _kill_code_executing_process(self) -> None:
        """
        Kills code executing process if it exists
        """
        if self._proc is not None:
            self._proc.kill()
            self._proc = None

    def update_df(self, df: pd.DataFrame | list[pd.DataFrame], df_original_name: str | list[str] | None) -> None:
        """
        Updates df(s) being used in the execution optionally together with their original names used for their potential saving.
        Forces new process creation and thus df copying.
        """
        self._df = df
        self._df_original_name = df_original_name
        self._start_new_code_executing_process()

    def update_functions_list(self, functions_list: list[Callable]) -> None:
        """
        Updates functions_list being used in the execution.
        Forces new process creation and thus df copying.
        """
        self._functions_list = functions_list
        self._start_new_code_executing_process()

    def _create_new_queues(self) -> None:
        """
        When new process is created, we also may need to create new queues
        """
        self._mp_input_queue = Queue()
        self._mp_output_queue = Queue()

    def _start_new_code_executing_process(self) -> None:
        """
        Creates and starts new process (while killing current process if it exists) for 
        code exection using  current `self._df` and `self._functions_list`
        """
        self._kill_code_executing_process()
        exec_kwargs, df_kwargs_keys = Code._create_exec_kwargs(
            self._df, self._df_original_name, self._functions_list)

        self._create_new_queues()
        self._proc = Process(
            target=run_code_execution_process,
            args=(exec_kwargs, df_kwargs_keys, self._mp_input_queue, self._mp_output_queue),
            daemon=True  # So it's killed automaticaly when this parent process gets killed
        )
        self._proc.start()

    def exec_code_with_timeout_in_separate_proc(self, code_str: str, timeout: int, save_df: bool, save_result_only: bool) -> tuple[ExecExitReason, str]:
        """
        Executes given code string while using `self._df` and `self._functions_list` as 
        execution kwargs and uses `timeout` for the timeout of exection

        Returns
        -------
        If the code terminates successfuly:
            Returns a tuple containing ExecExitReason.SUCCESS and std::out string generated by the code.
        If the code generates Exception:
            Returns a tuple containing ExecExitReason.EXCEPTION and execution's traceback given 
            by `Code.filter_exec_traceback` function
        If the code is terminated due to timeout
            Returns a tuple containing ExecExitReason.TIMEOUT and 'Program execution took too 
            long and thus was terminated.' string
        """

        # This starts code execution in new process
        self._mp_input_queue.put((
            code_str,
            self._df_saving_dir_path if save_df else None,
            save_result_only
        ))

        # Wait for `timeout` seconds while joining
        try:
            results = self._mp_output_queue.get(timeout=timeout)
            return results

        # If the execution has not finished in time
        except Exception:
            # Terminate the process and ready another one
            self._start_new_code_executing_process()
            return (
                ExecExitReason.TIMEOUT,
                'Program execution took too long and thus was terminated.',
                None
            )

    @staticmethod
    def execute_generated_code_no_thread(code_str: str, args: dict, tagged_query_type: str = "general", functions_list: list = []):
        try:
            args = {**args, **{f.__name__: f for f in functions_list}}
            for i in [1, 2]:
                with redirect_stdout(io.StringIO()) as output:
                    exec(code_str, globals(), args)
                results = copy.deepcopy(output.getvalue())

                if results != "" or tagged_query_type == "plot":
                    break
                # caused by no ```python  ``` in llm response or similar
                print(
                    f"{RED}{i}. Empty exec() output for the 'general' query type!{RESET}")
                if i == 2:
                    return "", "empty exec()"
            output.truncate(0)
            output.seek(0)
            return ExecExitReason.SUCCESS, results, args
        except Exception:
            exc_type, exc_value, tb = sys.exc_info()
            full_traceback = traceback.format_exc()
            # Filter the traceback
            exec_traceback = Code.filter_exec_traceback(
                full_traceback, exc_type.__name__, str(exc_value))
            print(f"{RED}   CODE PRODUCED AN ERROR{RESET}:\n\
                  {MAGENTA}     {exec_traceback}{RESET}\n")
            return ExecExitReason.EXCEPTION, exec_traceback, args

    @staticmethod
    def _create_exec_kwargs(df: pd.DataFrame | list[pd.DataFrame], df_original_name: str | list[str] | None, functions_list: list[Callable]) -> tuple[dict, dict[str, str]]:
        """
        Returns dictionary representing execution kwargs and dict mapping df_<num> names to their original ones (or the same ones if `df_original_name` was None).
        """
        # If original names were not supplied, we create adhoc ones.
        if df_original_name is None:
            df_original_name = 'df' if isinstance(df, pd.DataFrame) else [
                f'df_{i + 1}' for i in range(len(df))]

        if isinstance(df, pd.DataFrame):
            dfs_args = {'df': copy.deepcopy(df)}
            dfs_names = {'df': df_original_name}
        else:
            dfs_args = {
                f'df_{i + 1}': copy.deepcopy(cur_df) for i, cur_df in enumerate(df)}
            dfs_names = {f'df_{i + 1}': cur_df_original_name for i,
                         cur_df_original_name in enumerate(df_original_name)}
        
        # Add them so the `result` variable can be extracted
        dfs_args['result'] = None
        dfs_names['result'] = 'result'

        funcs_args = {f.__name__: f for f in functions_list}
        return {**dfs_args, **funcs_args}, dfs_names

    @staticmethod
    def _normalize_indentation(code_segment: str) -> str:
        # Determine the minimum indentation of non-empty lines
        lines = code_segment.strip().split('\n')
        min_indent = min(len(re.match(r'^\s*', line).group())
                         for line in lines if line.strip())

        # Remove the minimum indentation from each line
        return '\n'.join(line[min_indent:] for line in lines)

    # Method to clean the LLM response, and extract the code.
    # Method to clean the LLM response, and extract the code.
    @staticmethod
    def extract_code(response: str, provider: str, show_plot=False, prompt_strategy: str = "", model_name: str = "") -> str:
        # Use re.sub to replace all occurrences of the <|im_sep|> with the ```.
        response = re.sub(re.escape("<|im_sep|>"), "```", response)

        # Use a regular expression to find all code segments enclosed in triple backticks with "python"
        if provider == "local":
            code_segments = re.findall(
                r'```(?:python\s*)?(.*?)\s*```', response, re.DOTALL)
        else:
            if "```python" not in response and ('def solve' in response or 'print(' in response or 'import ' in response):
                response = "```python\n" + response + "\n```"

            code_segments = re.findall(
                r'```python\s*(.*?)\s*```', response, re.DOTALL)
            
        # Use a regular expression to find all code segments enclosed in triple backticks with or without "python"
        # print("code_segments:", code_segments)
        # code_segments = re.findall(r'```(?:python\s*)?(.*?)\s*```', response, re.DOTALL)
        if not code_segments:
            code_segments = re.findall(
                r'\[PYTHON\](.*?)\[/PYTHON\]', response, re.DOTALL)

        # Normalize the indentation for each code segment
        normalized_code_segments = [Code._normalize_indentation(
            segment) for segment in code_segments]

        return prompt_strategy.map(normalized_code_segments, Code._refine_extracted_code, {'provider': provider, 'show_plot': show_plot})

    def _refine_extracted_code(code: str, provider: str = "", show_plot=False):
        # Combine the normalized code segments into a single string
        # code_res = '\n'.join(normalized_code_segments).lstrip()
        code_res = code

        # Define a blacklist of Python keywords and functions that are not allowed
        blacklist = ['subprocess', 'sys', 'eval', 'exec', 'socket', 'urllib',
                     'shutil', 'pickle', 'ctypes', 'multiprocessing', 'tempfile', 'glob', 'pty',
                     'commands', 'cgi', 'cgitb',
                     'xml.etree.ElementTree', 'builtins', 'subprocess', 'sys', 'eval', 'exec', 'socket',
                     'urllib', 'shutil', 'pickle',
                     'ctypes', 'multiprocessing', 'tempfile', 'glob', 'pty', 'commands', 'cgi', 'cgitb',
                     'xml.etree.ElementTree', 'builtins', 'os.system', 'os.popen', 'sys.modules',
                     '__import__', 'getattr', 'setattr', 'pickle.loads', 'execfile', 'exec', 'compile',
                     'input', 'ast.literal_eval'
                     ]  # TODO: add 'os'?

        # Remove any instances of "df = pd.read_csv('filename.csv')" from the code
        code_res = re.sub(r"df\s*=\s*pd\.read_csv\((.*?)\)", "", code_res)

        # This is necessary for local OS models, as they are not as good as OpenAI models deriving the instruction from the promt
        if provider == "local":
            # Replace all occurrences of "data" with "df" if "data=pd." is present
            if re.search(r"data=pd\.", code_res):
                code_res = re.sub(r"\bdata\b", "df", code_res)
            # Comment out the df instantiation if it is present in the generated code
            code_res = re.sub(r"(?<![a-zA-Z0-9_-])df\s*=\s*pd\.DataFrame\((.*?)\)",
                              "# The dataframe df has already been defined", code_res)

        if not show_plot:
            if "plt.show()" in code_res:
                print(f"{RED}PLT.SHOW() in the code!!!{RESET}:")
            code_res = code_res.replace("plt.show()", "")

        # Define the regular expression pattern to match the blacklist items
        pattern = r"^(.*\b(" + "|".join(blacklist) + r")\b.*)$"

        # Replace the blacklist items with comments
        code_res = re.sub(pattern, r"# not allowed \1",
                          code_res, flags=re.MULTILINE)

        # Return the cleaned and extracted code
        return code_res.strip()

    @staticmethod
    def filter_exec_traceback(full_traceback, exception_type, exception_value):
        # Split the full traceback into lines and filter those that originate from "<string>"
        filtered_tb_lines = [line for line in full_traceback.split(
            '\n') if '<string>' in line]

        # Combine the filtered lines and append the exception type and message
        filtered_traceback = '\n'.join(filtered_tb_lines)
        if filtered_traceback:  # Add a newline only if there's a traceback to show
            filtered_traceback += '\n'
        filtered_traceback += f"{exception_type}: {exception_value}"

        return filtered_traceback

    @staticmethod
    def _prepend_imports(code_str: str) -> str:
        return f"import pandas as pd\nimport matplotlib.pyplot as plt\nimport numpy as np\nfrom numbers import Number\n\n{code_str}"


    # @staticmethod
    # def _append_result_storage(code_str: str, n_dfs: int=None) -> str:
    #     if n_dfs is None or n_dfs <= 1:
    #         return code_str + "\n\n" + "result = solve(df)\nprint(result)"
    #     else:
    #         return code_str + "\n\n" + f"result = solve({', '.join([f'df_{i+1}' for i in range(n_dfs)])})\nprint(result)"

    @staticmethod
    def remove_result_storage_lines(code_str: str) -> str:
        return re.sub(r"result = solve\(df\)\nprint\(result\)", "", code_str)

    @staticmethod
    def _format_to_pep8(code_str: str) -> str:
        # Removes redundant whitespaces and formats the code to PEP8
        return autopep8.fix_code(code_str)

    @staticmethod
    def preprocess_extracted_code(extracted_code: str, prompt_strategy: BasePromptStrategy, n_dfs: int = None) -> str:
        if "import pandas as pd" not in extracted_code or "import matplotlib.pyplot as plt" not in extracted_code:
            extracted_code = Code._prepend_imports(extracted_code)
        extracted_code = prompt_strategy.append_to_code(
            extracted_code, n_dfs=n_dfs)

        return Code._format_to_pep8(extracted_code)

    def exec_code_in_this_proc(self, code_str: str, save_df: bool, save_result_only: bool) -> Optional[tuple[ExecExitReason, str]]:
        """
        Executes given code string while using `self._df` and `self._functions_list` as execution kwargs in current process.
        WARNING: May never return...
        """

        with redirect_stdout(io.StringIO()) as output:
            # Try to run the code and let its std::out be noted in output
            try:
                exec_kwargs, df_kwargs_keys = Code._create_exec_kwargs(self._df, self._df_original_name, self._functions_list)
                exec(code_str, exec_kwargs)

                if save_df:
                    save_dfs_from_exec_kwargs(exec_kwargs, df_kwargs_keys, self._df_saving_dir_path, save_result_only)
                    saved_dir = self._df_saving_dir_path
                else:
                    saved_dir = None

            # In case code execution dies we return the exception as string
            except Exception:
                exc_type, exc_value, tb = sys.exc_info()
                full_traceback = traceback.format_exc()
                exec_traceback = Code.filter_exec_traceback(
                    full_traceback, exc_type.__name__, str(exc_value))

                return (
                    ExecExitReason.EXCEPTION,
                    exec_traceback,
                    saved_dir
                )

        # If no Exception was caught then return the program's output
        return (
            ExecExitReason.SUCCESS,
            output.getvalue(),
            saved_dir
        )

    def execute_generated_code(self, code_str: str, tagged_query_type: str = "", timeout: int = 100, save_df: bool = False, same_proc_exec: bool = False, save_result_only: bool = True) -> tuple:
        """
        The main method by which the agent should execute its code.

        Parameters
        ----------
        save_result_only: bool = True
            If `save_df` is true, then it determines whether only the result variable does get saved or all of them (dataframes + result) do.
        """
        print(f"{BLUE}EXECUTING THE CODE{RESET}:")
        for i in [1, 2]:

            t_start = time()

            if same_proc_exec is True and timeout is None:
                exec_exit_reason, result, path_to_saved_dfs = self.exec_code_in_this_proc(code_str, save_df, save_result_only)
            else:
                exec_exit_reason, result, path_to_saved_dfs = self.exec_code_with_timeout_in_separate_proc(code_str, timeout, save_df, save_result_only)

            t_elapsed = time() - t_start

            # If the generated code finished its execution
            if exec_exit_reason is ExecExitReason.SUCCESS:

                # And furthermore the generated result isn't trivialy incorrect
                if result != '' or tagged_query_type == 'plot':
                    exec_traceback = result
                    print(f"{YELLOW}   FINISHED EXECUTING, RESULTS\
                          {MAGENTA}:\n     {result}{RESET}\n")
                    return result, exec_traceback, exec_exit_reason, t_elapsed, path_to_saved_dfs

                # In case its obviously wrong
                else:
                    # Print such information
                    print(f"{RED}{i}. Empty exec() output for the 'general' query type!\
                          {RESET}")  # caused by no ```python  ``` in gpt's response?
                    # If it was the last round return this error
                    if i == 2:
                        return '', 'empty exec()', exec_exit_reason, t_elapsed, path_to_saved_dfs
            elif exec_exit_reason is ExecExitReason.EXCEPTION:
                exec_traceback = result
                print(f"{RED}   CODE PRODUCED AN ERROR{RESET}:\n\
                      {MAGENTA}     {exec_traceback}{RESET}\n")
                return 'ERROR', exec_traceback, exec_exit_reason, t_elapsed, path_to_saved_dfs

            # Else it terminated on Excpetion or timeout the message of which
            # is in this case summarized for agent in `result`
            elif exec_exit_reason is ExecExitReason.TIMEOUT:
                exec_traceback = result
                print(f"{RED}   'PROGRAM EXECUTION TOOK TOO LONG AND THUS WAS TERMINATED.'\
                      {RESET}:\n{MAGENTA}     {exec_traceback}{RESET}\n")
                return 'TIMEOUT', exec_traceback, exec_exit_reason, t_elapsed, path_to_saved_dfs

            else:
                print(f"{RED}   'UNKNOWN EXECUTION EXIT REASON.'\
                      {RESET}:\n{MAGENTA}{RESET}\n")
