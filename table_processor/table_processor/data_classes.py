from dataclasses import dataclass


@dataclass
class BaseDataClass:

    def __getitem__(self, item):
        return getattr(self, item)
    

@dataclass
class CodeSnippet(BaseDataClass):

    code_segment: str | None = None
    final_code_segment: str | None = None
    result_of_execution: str | None = None
    code_errors: str | None = None
    plot_filenames: list[str] | None = None
    successful_code_execution: bool | None = None
    traceback: str | None = None
    path_to_saved_dfs: str | None = None
    exec_exit_reason: str | None = None
    last_debug_prompt: str | None = None
    wall_time_final_code_runtime: float | None = None
    wall_time_full: float | None = None
    count_of_fixing_errors: int | None = None
    query_type: str | None = None # general or plot

    def __getitem__(self, item):
        # legacy reasons:
        if item == 'generated_code':
            return self.code_segment
        elif item == 'final_generated_code':
            return self.final_code_segment
        elif item == 'result_repl_stdout':
            return self.result_of_execution
        elif item == 'code_errors':
            return self.code_errors
        elif item == 'plot_filename':
            return self.plot_filenames
        elif item == 'successful_code_execution':
            return self.successful_code_execution
        elif item == 'traceback':
            return self.traceback
        elif item == 'path_to_saved_dfs':
            return self.path_to_saved_dfs
        elif item == 'exec_exit_reason':
            return self.exec_exit_reason
        elif item == 'last_debug_prompt':
            return self.last_debug_prompt
        elif item == 'code_execution_was_required':
            return True
        elif item == 'first_generated_code':
            return self.code_segment
        else:
            return getattr(self, item)