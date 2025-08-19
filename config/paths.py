"""
Cross-platform path configuration for the customer support system.
Works on Windows, Linux, and Docker containers.
"""

import os
from pathlib import Path
from typing import Dict

# Get the project root directory (where main.py is located)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories configuration
DATA_ROOT = PROJECT_ROOT / "data"
FILES_PROCESSED_ROOT = DATA_ROOT / "files_processed"

# Specialist data paths
SPECIALIST_DATA_PATHS: Dict[str, Path] = {
    'equipment': FILES_PROCESSED_ROOT / "equipment",
    'cables': FILES_PROCESSED_ROOT / "cables", 
    'tools': FILES_PROCESSED_ROOT / "tools",
    'common_info': FILES_PROCESSED_ROOT / "common_info"
}

# Convert to strings for compatibility with existing code
SPECIALIST_DATA_PATHS_STR: Dict[str, str] = {
    key: str(path) for key, path in SPECIALIST_DATA_PATHS.items()
}

def get_data_path(specialist: str) -> Path:
    """
    Get the data path for a specific specialist.
    
    Args:
        specialist: Name of the specialist ('equipment', 'cables', 'tools', 'common_info')
        
    Returns:
        Path: Absolute path to the specialist's data directory
        
    Raises:
        KeyError: If specialist name is not recognized
    """
    if specialist not in SPECIALIST_DATA_PATHS:
        raise KeyError(f"Unknown specialist: {specialist}. Available: {list(SPECIALIST_DATA_PATHS.keys())}")
    
    return SPECIALIST_DATA_PATHS[specialist]

def get_data_path_str(specialist: str) -> str:
    """
    Get the data path for a specific specialist as a string.
    
    Args:
        specialist: Name of the specialist
        
    Returns:
        str: Absolute path to the specialist's data directory as string
    """
    return str(get_data_path(specialist))

def ensure_data_directories():
    """
    Ensure all data directories exist. Create them if they don't.
    Useful for Docker containers or fresh installations.
    """
    for path in SPECIALIST_DATA_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

def list_data_files(specialist: str, pattern: str = "*.txt") -> list:
    """
    List all files matching pattern in specialist's data directory.
    
    Args:
        specialist: Name of the specialist
        pattern: File pattern to match (default: "*.txt")
        
    Returns:
        list: List of matching file paths
    """
    data_path = get_data_path(specialist)
    return list(data_path.glob(pattern))

def get_image_path(specialist: str, doc_name: str, image_name: str) -> Path:
    """
    Get path to a specific image file.
    
    Args:
        specialist: Name of the specialist
        doc_name: Document name (without extension)
        image_name: Image filename
        
    Returns:
        Path: Path to the image file
    """
    return get_data_path(specialist) / doc_name / image_name

def get_mapping_file_path(specialist: str, mapping_type: str) -> Path:
    """
    Get path to a mapping file (pdf_mapping.json, doc_mapping.json, etc.).
    
    Args:
        specialist: Name of the specialist
        mapping_type: Type of mapping file ('pdf_mapping.json', 'doc_mapping.json', etc.)
        
    Returns:
        Path: Path to the mapping file
    """
    return get_data_path(specialist) / mapping_type

def get_pdf_mapping_path(specialist: str) -> Path:
    """
    Get path to pdf_mapping.json for a specialist.
    
    Args:
        specialist: Name of the specialist
        
    Returns:
        Path: Path to the pdf_mapping.json file
    """
    return get_mapping_file_path(specialist, 'pdf_mapping.json')

def get_doc_mapping_path(specialist: str) -> Path:
    """
    Get path to doc_mapping.json for a specialist.
    
    Args:
        specialist: Name of the specialist
        
    Returns:
        Path: Path to the doc_mapping.json file
    """
    return get_mapping_file_path(specialist, 'doc_mapping.json')

# Legacy compatibility - for existing code that expects string paths
PATHS = SPECIALIST_DATA_PATHS_STR

# Environment-specific adjustments
def get_runtime_info():
    """Get information about the current runtime environment."""
    return {
        'is_docker': os.path.exists('/.dockerenv'),
        'platform': os.name,
        'project_root': str(PROJECT_ROOT),
        'data_root': str(DATA_ROOT)
    }

# Initialize directories on import (safe for both Windows and Docker)
try:
    ensure_data_directories()
except Exception:
    # Silently fail if we don't have write permissions
    pass