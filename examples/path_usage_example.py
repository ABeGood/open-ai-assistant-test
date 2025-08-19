"""
Example showing how to use the new cross-platform path system
with existing agent response processing utilities.
"""

import sys
import os

# Add project root to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_response_processing_utils import process_image_markers, extract_marker_parts
from agents.path_utils import resolve_image_path, resolve_all_images_in_text, find_specialist_files
from config import get_runtime_info, ensure_data_directories


def example_response_processing():
    """Example of processing a response with image markers."""
    
    # Sample response text with image markers
    response_text = """
    The MS021 equipment setup is shown in @D_3_IMG_002_setup.
    For troubleshooting, refer to @D_3_IMG_005_diagram and @D_3_IMG_007_flowchart.
    The final connection should look like @D_3_IMG_010_final.
    """
    
    print("=== Original Response ===")
    print(response_text)
    
    # Process image markers (existing function)
    cleaned_text, markers = process_image_markers(response_text)
    
    print("\n=== Processed Response ===")
    print(f"Cleaned text: {cleaned_text}")
    print(f"Extracted markers: {markers}")
    
    # Resolve image paths using new cross-platform system
    specialist = "equipment"  # This response is from equipment specialist
    image_paths = resolve_all_images_in_text(specialist, markers)
    
    print(f"\n=== Resolved Image Paths ===")
    for marker, path in zip(markers, image_paths):
        if path:
            print(f"{marker} -> {path}")
        else:
            print(f"{marker} -> [NOT FOUND]")
    
    return cleaned_text, [p for p in image_paths if p]  # Return only found paths


def example_file_discovery():
    """Example of discovering files in specialist directories."""
    
    print("\n=== File Discovery Example ===")
    
    specialists = ["equipment", "cables", "tools", "common_info"]
    
    for specialist in specialists:
        print(f"\n{specialist.upper()} Files:")
        try:
            txt_files = find_specialist_files(specialist, "*.txt")
            print(f"  Found {len(txt_files)} .txt files")
            if txt_files:
                # Show first few files as examples
                for file_path in txt_files[:3]:
                    filename = os.path.basename(file_path)
                    print(f"    - {filename}")
                if len(txt_files) > 3:
                    print(f"    ... and {len(txt_files) - 3} more")
        except Exception as e:
            print(f"  Error: {e}")


def example_runtime_detection():
    """Example of detecting runtime environment."""
    
    print("\n=== Runtime Environment ===")
    runtime_info = get_runtime_info()
    
    for key, value in runtime_info.items():
        print(f"{key}: {value}")
    
    # Adjust behavior based on environment
    if runtime_info['is_docker']:
        print("\n🐳 Running in Docker container")
        print("  - Using container-optimized paths")
        print("  - Data persistence may require volume mounts")
    else:
        print(f"\n💻 Running on {runtime_info['platform']}")
        print("  - Using local file system paths")


def example_individual_image_resolution():
    """Example of resolving individual image markers."""
    
    print("\n=== Individual Image Resolution ===")
    
    test_markers = [
        "@D_3_IMG_002",
        "@D_15_IMG_005", 
        "@D_21_IMG_010",
        "@D_999_IMG_001"  # This one shouldn't exist
    ]
    
    for marker in test_markers:
        print(f"\nTesting marker: {marker}")
        
        # Extract parts
        parts = extract_marker_parts(marker)
        if parts:
            print(f"  Parsed: {parts}")
            
            # Try to resolve for equipment specialist
            path = resolve_image_path("equipment", marker)
            if path:
                print(f"  Resolved: {path}")
                print(f"  Exists: {os.path.exists(path)}")
            else:
                print(f"  Not found in equipment data")
        else:
            print(f"  Invalid marker format")


if __name__ == "__main__":
    print("Cross-Platform Path System Example")
    print("=" * 50)
    
    # Ensure data directories exist (safe in both Windows and Docker)
    ensure_data_directories()
    
    # Run examples
    example_runtime_detection()
    example_response_processing()
    example_file_discovery()
    example_individual_image_resolution()
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nThis code will work on:")
    print("  ✓ Windows (development)")
    print("  ✓ Linux (production)")  
    print("  ✓ Docker containers")
    print("  ✓ Any platform with Python pathlib support")