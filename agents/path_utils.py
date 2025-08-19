"""
Path utilities that integrate with agent_response_processing_utils.py
Provides cross-platform path resolution for Windows and Docker.
"""

import os
import glob
from pathlib import Path
from typing import List, Optional

from config import get_data_path, get_image_path, get_pdf_mapping_path, get_doc_mapping_path, SPECIALIST_DATA_PATHS_STR


def resolve_image_path(specialist: str, marker: str) -> Optional[str]:
    """
    Resolve an image marker to an actual file path.
    Works on both Windows and Docker containers.
    
    Args:
        specialist: Name of the specialist ('equipment', 'cables', 'tools', 'common_info')
        marker: Image marker like "@D_3_IMG_002"
        
    Returns:
        str: Full path to the image file, or None if not found
        
    Example:
        >>> resolve_image_path('equipment', '@D_3_IMG_002')
        '/app/data/files_processed/equipment/MS021_UM_ENG/D_3_IMG_002.jpeg'
    """
    # Import here to avoid circular imports
    from .agent_response_processing_utils import extract_marker_parts
    
    marker_parts = extract_marker_parts(marker)
    if not marker_parts:
        return None
    
    img_file_key = marker_parts["img_file_key"]  # e.g., "D_3"
    img_name = marker_parts["img_name"]          # e.g., "IMG_002"
    
    # Get the base data path for the specialist
    base_path = get_data_path(specialist)
    
    # Look for image files in subdirectories
    # Pattern: specialist_path/*/{img_file_key}_{img_name}.*
    search_pattern = base_path / "*" / f"{img_file_key}_{img_name}.*"
    
    # Use glob to find matching files
    matching_files = list(base_path.glob(f"*/{img_file_key}_{img_name}.*"))
    
    if matching_files:
        return str(matching_files[0])  # Return first match
    
    return None


def find_specialist_files(specialist: str, pattern: str = "*.txt") -> List[str]:
    """
    Find all files matching pattern in specialist's directory.
    
    Args:
        specialist: Name of the specialist
        pattern: File pattern to match (default: "*.txt")
        
    Returns:
        List[str]: List of matching file paths as strings
    """
    from config import list_data_files
    
    file_paths = list_data_files(specialist, pattern)
    return [str(path) for path in file_paths]


def get_specialist_base_path(specialist: str) -> str:
    """
    Get the base path for a specialist's data directory as string.
    
    Args:
        specialist: Name of the specialist
        
    Returns:
        str: Absolute path to specialist's data directory
    """
    return SPECIALIST_DATA_PATHS_STR.get(specialist, "")


def resolve_all_images_in_text(specialist: str, markers: List[str]) -> List[str]:
    """
    Resolve multiple image markers to actual file paths.
    
    Args:
        specialist: Name of the specialist
        markers: List of image markers like ["@D_3_IMG_002", "@D_3_IMG_003"]
        
    Returns:
        List[str]: List of resolved file paths (empty strings for unresolved markers)
    """
    resolved_paths = []
    
    for marker in markers:
        path = resolve_image_path(specialist, marker)
        resolved_paths.append(path if path else "")
    
    return resolved_paths


def ensure_cross_platform_path(path_str: str) -> str:
    """
    Ensure path string works on both Windows and Unix systems.
    
    Args:
        path_str: Path string that might have platform-specific separators
        
    Returns:
        str: Normalized path string
    """
    return str(Path(path_str))


# Legacy compatibility functions for existing code
def get_equipment_path() -> str:
    """Get equipment data path (legacy compatibility)."""
    return SPECIALIST_DATA_PATHS_STR['equipment']


def get_cables_path() -> str:
    """Get cables data path (legacy compatibility)."""
    return SPECIALIST_DATA_PATHS_STR['cables']


def get_tools_path() -> str:
    """Get tools data path (legacy compatibility)."""
    return SPECIALIST_DATA_PATHS_STR['tools']


def get_common_info_path() -> str:
    """Get common_info data path (legacy compatibility)."""
    return SPECIALIST_DATA_PATHS_STR['common_info']


def get_pdf_mapping_file_path(specialist: str) -> str:
    """
    Get the path to pdf_mapping.json for a specialist as a string.
    This fixes the path concatenation issue that causes missing path separators.
    
    Args:
        specialist: Name of the specialist
        
    Returns:
        str: Full path to the pdf_mapping.json file
        
    Example:
        >>> get_pdf_mapping_file_path('equipment')
        'D:\\Projects\\open-ai-assistant-test\\data\\files_processed\\equipment\\pdf_mapping.json'
    """
    return str(get_pdf_mapping_path(specialist))


def get_doc_mapping_file_path(specialist: str) -> str:
    """
    Get the path to doc_mapping.json for a specialist as a string.
    
    Args:
        specialist: Name of the specialist
        
    Returns:
        str: Full path to the doc_mapping.json file
    """
    return str(get_doc_mapping_path(specialist))


def get_mapping_file_path_safe(specialist: str, filename: str) -> str:
    """
    Safely construct a mapping file path with proper path separators.
    This replaces the unsafe string concatenation in existing code.
    
    Args:
        specialist: Name of the specialist
        filename: The mapping filename (e.g., 'pdf_mapping.json')
        
    Returns:
        str: Full path to the mapping file with proper separators
        
    Example:
        >>> get_mapping_file_path_safe('equipment', 'pdf_mapping.json')
        'D:\\Projects\\open-ai-assistant-test\\data\\files_processed\\equipment\\pdf_mapping.json'
    """
    from config import get_mapping_file_path
    return str(get_mapping_file_path(specialist, filename))