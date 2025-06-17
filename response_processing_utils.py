import os
import re
import glob
import json

assistant_files_mapping = {
    'equipment': 'files_preproc/equipment/',
    'cables': 'files_preproc/cables/',
    'tools': 'files_preproc/tools/',
    'commonInfo': 'files_preproc/common_info/'
}

IMG_PATTERN = r"D_\d+_IMG_\d+"

def get_all_markers_as_list(text: str) -> list[str]:
    """
    Finds all occurrences of the @_IMG_NNN pattern in a text
    and returns them as a list of strings.

    Args:
        text: The input string to search.

    Returns:
        A list of strings, where each string is a detected marker.
        Returns an empty list if no markers are found.
    """
    matches = re.findall(IMG_PATTERN, text)
    return matches

def remove_all_markers(text: str) -> str:
    """
    Removes all occurrences of the @D_N_IMG_NNN pattern from a text.

    Args:
        text: The input string to clean.

    Returns:
        A string with all markers removed.
        Returns the original string if no markers are found.
    """
    cleaned_text = re.sub(IMG_PATTERN, "", text)
    return cleaned_text

def extract_marker_parts(marker: str) -> dict[str, str] | None:
    """
    Extracts the two main components from a marker string.
    
    Args:
        marker: A marker string like "@D_3_IMG_002"
        
    Returns:
        A tuple containing (prefix_part, img_part), e.g., ("D_3", "IMG_002")
        Returns None if the marker doesn't match the expected format.
    """
    # Pattern with two capturing groups
    pattern = r"(D_\d+)_(IMG_\d+)"
    
    match = re.match(pattern, marker)
    if match:
        return {
            "img_file_key": match.group(1), 
            "img_name": match.group(2)
        }
    return None

def clean_message_text(text):
    """Clean text by removing problematic characters"""
    # Remove backslashes
    text = text.replace("\\", "")
    
    # Remove other potentially problematic characters
    problematic_chars = ['*', '_', '[', ']', '~', '`', '>', '#', '=', '|', '{', '}']
    for char in problematic_chars:
        text = text.replace(char, "")
    
    # Clean up whitespace
    text = " ".join(text.split())
    
    return text

def delete_sources_from_text(text: str):
    pattern = r'【.*?】'
    return re.sub(pattern, '', text)

def find_file_by_name(folder_path: str, filename_base: str) -> list[str]:
    """
    Finds all files with the given base name regardless of extension.
    
    Args:
        folder_path: Path to the folder to search in
        filename_base: Base filename without extension (e.g., "IMG_001")
        
    Returns:
        List of full file paths that match the base name
    """
    # Create search pattern
    search_pattern = os.path.join(folder_path, f"{filename_base}.*")
    
    # Find all matching files
    matching_files = glob.glob(search_pattern)
    
    return matching_files
