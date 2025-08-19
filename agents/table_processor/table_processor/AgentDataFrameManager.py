import os
import json
import copy
import pathlib
from enum import Enum
from typing import Callable

import pandas as pd

from .llms import LLM
from .Observer import Publisher



class AddDataMode(Enum):
    APPEND       = 1
    CHECK_APPEND = 2
    CHECK_RELOAD = 3




class AgentDataFrameManager(Publisher):
    """
    Provides friendly interface for AgentDataFrame-s.

    Notifies for topics:
        - df_change
    """

    def __init__(
            self,
            table_file_paths: list[str] | str | None = None,
            data_specs_dir_path: str | None = None,
            llm_calls: LLM | None = None,
            forced_data_specs: list[dict | None] | dict | None = None
        ) -> None:
        """
        Creates AgentDataFrameManager.

        Parameters:
        -----------
        table_file_paths: list[str] | str | None
            Either list of file paths, single file path specifying from where the data should be loaded 
            or None in case of initializing empty.

        data_specs_dir_path: str | None
            Path to the directory where we look for annotations files. If annotations 
            are not found in the directory, then we create them by using `llm_calls`.

            If it is set to None, then no annotations are being used (And thus not even generated).

        llm_calls: LLM | None
            LLM class instance which is required to supply when `data_specs_dir_path` is not None, 
            as it may be used for generating data specifications if they are not found in `data_specs_dir_path`
        """
        super().__init__()
        self._data_specs_dir_path = data_specs_dir_path
        self._lmm_calls = llm_calls
        self._data: list[AgentDataFrame] = []

        # Load the possibly supplied data
        if table_file_paths is not None:
            self.add_data(table_file_paths, forced_annotations=forced_data_specs)

    def _append_add_data(self, table_file_path: str, forced_annotations: dict | None = None) -> None:
        """
        Performs `self.add_data` in `AddDataMode.APPEND` mode.
        """
        self._data.append(
            AgentDataFrame(
                table_file_path,
                self._data_specs_dir_path,
                self._lmm_calls,
                forced_data_specs=forced_annotations
            )
        )
    
    def _get_index_of_equaly_sourced_data(self, table_file_path: str) -> int | None:
        """
        Returns index within `self._data` of AgentDataFrame which is loaded from the same
        source file. If no such AgentDataFrame instance exists None is returned.
        """
        for i, tfp in enumerate(self._data):
            if tfp.get_table_file_path() == table_file_path:
                return i
            
        return None
        
    def add_data(self, table_file_paths: list[str] | str, forced_annotations: list[dict | None] | dict | None = None, add_data_mode: AddDataMode = AddDataMode.APPEND) -> None:
        """
        Loads dataframe(s) from given location(s)

        Parameters:
        -----------
        table_file_paths: list[str] | str
            file path or list of file paths to load dataframes from  

        forced_annotations: list[str | None] | str | None, default = None
            Annotations which would be forcefully tied to the given table file path(s) 
            thus overforcing annotations which would be otherwise loaded or LLM generated.
        
        add_data_mode: AddDataMode = AddDataMode.APPEND
            Mode defining the behaviour. There are 3 options:
                
                . AddDataMode.APPEND
                    Simply loads and appends the data to the list.
                
                . AddDataMode.CHECK_APPEND
                    Checks whether data sourced from the same file are already loaded.
                    If they are found nothing is done. If not then it is appended.    

                    Useful when we want to add data of which we don't want duplicate and
                    we are sure the original was not altered

                . AddDataMode.CHECK_RELOAD
                    Checks whether data sourced from the same file are already loaded.
                    If they are found they are reloaded from path. If not then it is appended.

                    Useful when we want to add data of which we don't want duplicate and
                    we are not sure whether the original was altered or not.
        """
        # table_file_paths and forced_annotations validation and type conversion
        if isinstance(table_file_paths, str):
            table_file_paths = [table_file_paths]

        # If both are lists check their lengths
        if type(table_file_paths) is type(forced_annotations) is list:
            assert len(forced_annotations) == len(table_file_paths), 'The length of table_file_paths and forced_annotations must be the same. If you wish not to force annotations for some files pass None on the respective positions in forced_annotations.'

        # If both need to be changed to lists do so
        if type(table_file_paths) is str:             
            table_file_paths = [table_file_paths]
        if type(forced_annotations) is dict:
            forced_annotations = [forced_annotations]                        

        # If no annotations are to be forced, then we implement this simply by having list full of Nones as forced_annotations
        if forced_annotations is None or type(forced_annotations)==AddDataMode:
            forced_annotations = [None] * len(table_file_paths)

        # Now after working with input parameters let's use them
        for tfp, f_annot in zip(table_file_paths, forced_annotations):
            match add_data_mode.name:
            
                case AddDataMode.APPEND.name:
                    self._append_add_data(tfp, forced_annotations=f_annot)

                case AddDataMode.CHECK_APPEND.name:
                    if self._get_index_of_equaly_sourced_data(tfp) is None:
                        self._append_add_data(tfp, forced_annotations=f_annot)                    

                case AddDataMode.CHECK_RELOAD.name:                                    
                    if (idx := self._get_index_of_equaly_sourced_data(tfp)) is not None:
                        self._data.pop(idx)         

                    self._append_add_data(tfp, forced_annotations=f_annot)

                case _:
                    raise Exception('Invalid AddDataMode supplied to AgentDataFrameManager')
        
        self.notify('df_change')

    def remove_all_data(self) -> None:
        """
        Completely empties the list of stored data.
        """
        self._data = []
        self.notify('df_change')


    def remove_dataframe_and_annotation(self, index:int) -> None:
        """
        Non-Completely empties the list of stored data.
        """
        try:
            self._data.pop(index) 
            self.notify('df_change')
            print(f'Dataframe Manager list len: {len(self._data)}')
            return True
        except:
            return False
          
        


    def save_data_specs(self, save_dir: str) -> None:
        """
        Saves all data specifications to `save_dir`
        """
        for agent_data_frame in self._data:
            agent_data_frame.save_data_specs(save_dir)

    def save_dataframes(self, save_dir: str, file_format: str = '.csv') -> None:
        """
        Saves all dataframes (in given `file_format`) to `save_dir`
        """
        for agent_data_frame in self._data:
            agent_data_frame.save_dataframe(save_dir, file_format=file_format)
        

    def save_all_data(self, save_dir: str, file_format: str = '.csv') -> None:
        """
        Saves dataframes (in given `file_format`) aswell as data specifications to `save_dir`
        """
        self.save_data_specs(save_dir)
        self.save_dataframes(save_dir)

    def get_data_specs(self, return_copy: bool = False) -> list[dict] | dict | None:
        """
        Returns dict or list of dicts which represent data specs if they were to be loaded/generated.
        In opposite case it simply returns None.
        """
        # In case no specs were to be generated
        if self._data_specs_dir_path is None:
            return None
        
        # Else we have either loaded or generated specs for all data
        ret_data_specs = [adf.get_data_specs(return_copy=return_copy) for adf in self._data]

        return ret_data_specs
    
    def get_dataframes(self, return_copy: bool = False) -> list[pd.DataFrame] | pd.DataFrame:
        """
        Returns dataframe or list of dataframes which represent loaded data.    
        """
        # Get all of the loaded dataframes
        ret_dataframes = [adf.get_dataframe(return_copy=return_copy) for adf in self._data]

        return ret_dataframes[0] if len(ret_dataframes) == 1 else ret_dataframes
    
    def get_dataframes_source_filenames(self) -> list[str] | str:
        """
        Returns filename or list of filenames from which the dataframes were loaded.

        Favorable for standalone data saving and keeping the names which users know.
        """
        ret_filenames = [adf.get_filename() for adf in self._data]

        return ret_filenames[0] if len(ret_filenames) == 1 else ret_filenames
    
    def get_table_file_paths(self) -> list[str] | str:
        """
        Returns table file path or list of file paths from where the dataframes were loaded.
        """
        ret_table_file_paths = [adf.get_table_file_path() for adf in self._data]

        return ret_table_file_paths[0] if len(ret_table_file_paths) == 1 else ret_table_file_paths



class AgentDataFrame:
    """
    Class used for representing data and their potential annotations within the Agent.
    """

    def __init__(
            self,
            table_file_path: str,
            data_specs_dir_path: str | None,
            llm_calls: LLM | None,
            forced_data_specs: dict | None = None
        ) -> None:
        """
        Creates AgentDataFrame.

        Parameters:
        -----------
        table_file_path: str,
            path to the dataframe to load. Also defines the name of the data annotations files
            which are potentially being looked for in the `data_specs_dir_path`.
            
            Supports .csv, .parquet and .xlsx files.

        data_specs_dir_path: str | None
            Path to the directory where we look for annotations files. If annotations 
            are not found in the directory, then we create them by using `llm_calls`.
            If it is set to None, then no annotations are being used (And thus not even generated). 

            However if `forced_data_specs` is supplied then this parameter is overpowered and its value ignored.

        llm_calls: LLM | None
            LLM class instance which is required to supply when `data_specs_dir_path` is not None, 
            as it may be used for generating data specifications if they are not found in `data_specs_dir_path`

        forced_data_specs: dict | None = None
            If default None is kept then the data specifications are used as specified by parameter `data_specs_dir_path`. 
            However when given a dictionary it overpowers `data_specs_dir_path` parameter and forces given data specifications to be used.
        """
        self._table_file_path = table_file_path
        self._data_specs_dir_path = data_specs_dir_path
        self._llm_calls = llm_calls
        self._forced_data_specs = forced_data_specs

        self._load_data()

    def _load_df(self) -> None:
        """
        Loads dataframe from `self._table_file_path` to `self._df`        
        """
        file_format = pathlib.Path(self._table_file_path).suffix        
        match file_format:
        
            case '.csv':
                load_data_function = pd.read_csv

            case '.parquet':
                load_data_function = pd.read_parquet

            case '.xlsx':
                load_data_function = pd.read_excel

            case _:
                raise Exception(f'Unsupported file format {file_format} was expected to be loaded. Supported file formats are .csv, .parquet and .xlsx only.')

        self._df: pd.DataFrame = load_data_function(self._table_file_path)

    def _load_data_specs(self) -> None:
        """
        Loads data specifications into `self._data_specs_dir_path`
        """
        # In case of forced data specifications
        if self._forced_data_specs is not None:
            self._data_specs = self._forced_data_specs
            return

        # In case no specifications are to be used
        if self._data_specs_dir_path is None:
            self._data_specs = None
            return

        potential_file_name = os.path.join(self._data_specs_dir_path, f'{pathlib.Path(self._table_file_path).stem}.json')
        
        # Try to load the annotations file
        try:
            with open(potential_file_name) as f:
                self._data_specs = json.load(f)

        # If the annotations do not exist, create them
        except:
            columns = [
                {
                    'name': col, 
                    'type': str(self._df[col].dtype)
                } 
                for col in self._df.columns
            ]

            table_name = os.path.splitext(self._filename)[0]
            generated_descriptions = self._llm_calls.generate_column_descriptions(table_name, columns)

            generated_descriptions_dict = json.loads(generated_descriptions)

            # Create a dictionary to map column names to their descriptions and units
            description_map = {desc["name"]: desc["description"] for desc in generated_descriptions_dict}
            unit_map = {desc["name"]: desc["unit"] for desc in generated_descriptions_dict}

            # Add descriptions and units to the columns
            for col in columns:
                col["description"] = description_map.get(col["name"], "")
                col["unit"] = unit_map.get(col["name"], None)

            self._data_specs = {
                "table_name": table_name,
                "description": f"Table containing data for {table_name}",
                "columns": columns
            }

    def _load_data(self) -> None:
        """
        Loads all data to the instance.
        """
        self._filename = pathlib.Path(self._table_file_path).stem
        self._load_df()
        self._load_data_specs()  

    def save_data(self, save_dir: str, file_format: str = '.csv') -> None:
        """
        Saves data into `save_dir` while saving `self._df` as `file_format`.
        """ 
        self.save_dataframe(save_dir, file_format=file_format)
        self.save_data_specs(save_dir)

    def get_dataframe(self, return_copy: bool = False) -> pd.DataFrame:
        """
        Returns loaded dataframe.
        """
        return copy.deepcopy(self._df) if return_copy else self._df
    
    def save_dataframe(self, save_dir: str, file_format: str = '.csv') -> None:
        """
        Saves `self._df` into `save_dir` using `file_format`.
        """        
        match file_format:
        
            case '.csv':
                save_data_function = pd.DataFrame.to_csv

            case '.parquet':
                save_data_function = pd.DataFrame.to_parquet

            case '.xlsx':
                save_data_function = pd.DataFrame.to_excel

            case _:
                raise Exception(f'Unsupported file format {file_format} was expected to be saved as. Supported file formats are .csv, .parquet and .xlsx only.')
        
        os.makedirs(save_dir, exist_ok=True)
        save_data_function(
            self._df,
            os.path.join(save_dir, f'{self.get_filename()}{file_format}')
        )
    
    
    def get_data_specs(self, return_copy: bool = False) -> dict | None:
        """
        Returns possibly loaded annotations.
        """
        return copy.deepcopy(self._data_specs) if return_copy else self._data_specs
    
    def save_data_specs(self, save_dir: str) -> None:
        """
        Saves `self._data_specs` if they are not None into `save_dir`.
        """
        if self._data_specs is not None:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'{self.get_filename()}.json')

            with open(path, 'w') as f:
                json.dump(self._data_specs, f)

    def get_filename(self) -> str:
        """
        Returns filename from which the `self._df` was loaded.
        """
        return self._filename
    
    def get_table_file_path(self) -> str:
        """
        Returns table file path from where the dataframe was loaded.
        """
        return self._table_file_path