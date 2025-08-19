import os
import re
import glob
import json

def process_image_markers(text: str) -> tuple[str, list[str]]:
    """
    Extracts image markers from text and replaces them with sequential figure references.
    
    Args:
        text: The input string containing image markers in format @D_N_IMG_NNN_...
        
    Returns:
        A tuple containing:
        - cleaned_text: Text with markers replaced by "Fig. 1", "Fig. 2", etc.
        - markers_list: List of core markers (@D_N_IMG_NNN) in order of appearance
    """
    # Pattern to capture full marker including @ and content until space
    FULL_IMG_PATTERN = r"@D_\d+_IMG_\d+_[^\s]*"
    
    # Pattern to extract core marker part
    CORE_IMG_PATTERN = r"@D_\d+_IMG_\d+"
    
    # Find all full markers in order of appearance
    full_markers = re.findall(FULL_IMG_PATTERN, text)
    
    # Extract core parts for the return list
    markers = []
    for full_marker in full_markers:
        core_match = re.search(CORE_IMG_PATTERN, full_marker)
        if core_match:
            markers.append(core_match.group())
    
    # Create a copy of text for processing
    cleaned_text = text
    
    # Replace each full marker with sequential figure reference
    for i, full_marker in enumerate(full_markers, 1):
        # Replace only the first occurrence to maintain order
        cleaned_text = cleaned_text.replace(full_marker, f"Fig. {i}", 1)
    
    return cleaned_text, markers

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
    pattern = r"@(D_\d+)_(IMG_\d+)"
    
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
