"""
Configuration package for the customer support system.
"""

from .paths import (
    PROJECT_ROOT,
    DATA_ROOT,
    SPECIALIST_DATA_PATHS,
    SPECIALIST_DATA_PATHS_STR,
    get_data_path,
    get_data_path_str,
    ensure_data_directories,
    list_data_files,
    get_image_path,
    get_mapping_file_path,
    get_pdf_mapping_path,
    get_doc_mapping_path,
    get_runtime_info,
    PATHS  # Legacy compatibility
)

__all__ = [
    'PROJECT_ROOT',
    'DATA_ROOT', 
    'SPECIALIST_DATA_PATHS',
    'SPECIALIST_DATA_PATHS_STR',
    'get_data_path',
    'get_data_path_str',
    'ensure_data_directories',
    'list_data_files',
    'get_image_path',
    'get_mapping_file_path',
    'get_pdf_mapping_path',
    'get_doc_mapping_path',
    'get_runtime_info',
    'PATHS'
]