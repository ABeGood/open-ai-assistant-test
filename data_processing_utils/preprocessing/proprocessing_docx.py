from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
import os
from pathlib import Path
import json
from unidecode import unidecode

def extract_and_map_images(document_path, output_image_dir, doc_prefix=""):
    document = Document(document_path)
    image_map = {} # Maps rId to new filename
    
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    image_count = 0
    for rel in document.part.rels:
        if "image" in document.part.rels[rel].reltype:
            image_part = document.part.rels[rel].target_part
            image_bytes = image_part.blob
            image_extension = image_part.content_type.split('/')[-1]
            
            new_filename = f"{doc_prefix}_IMG_{image_count:03d}.{image_extension}"
            output_path = os.path.join(output_image_dir, new_filename)
            
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            
            image_map[rel] = new_filename # Store the rId -> new_filename mapping
            image_count += 1
            
    return document, image_map

def extract_text_from_element(element, image_map):
    """
    Recursively extract text from an element, handling hyperlinks and images.
    This function properly handles nested text elements including hyperlinks.
    """
    text_parts = []
    
    # Define namespace prefixes
    w_ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    a_ns = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
    r_ns = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
    
    # Check for direct text elements
    for t in element.findall(f'{w_ns}t'):
        if t.text:
            text_parts.append(t.text)
    
    # Check for hyperlinks (this is what was missing!)
    for hyperlink in element.findall(f'{w_ns}hyperlink'):
        # Recursively extract text from within hyperlinks
        hyperlink_text = extract_text_from_element(hyperlink, image_map)
        text_parts.append(hyperlink_text)
    
    # Check for runs within this element
    for run in element.findall(f'{w_ns}r'):
        # Extract text from text runs
        for t in run.findall(f'{w_ns}t'):
            if t.text:
                text_parts.append(t.text)
        
        # Check for drawings (images) within the run
        for drawing in run.findall(f'{w_ns}drawing'):
            blip_elements = drawing.findall(f'.//{a_ns}blip')
            for blip in blip_elements:
                r_id = blip.get(f'{r_ns}embed')
                if r_id in image_map:
                    marker_name = image_map[r_id].split('.')[0]
                    text_parts.append(f"@{marker_name}")
                    break
    
    return "".join(text_parts)

def convert_docx_to_plain_text_with_markers(document, image_map):
    """
    Enhanced version that properly handles hyperlinks and other nested text elements.
    """
    plain_text_lines = []
    
    # Define namespace prefix
    w_ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

    for block in document.element.body:
        if block.tag.endswith('p'): # It's a paragraph
            paragraph_text = extract_text_from_element(block, image_map)
            plain_text_lines.append(paragraph_text)

        elif block.tag.endswith('tbl'): # It's a table
            for row_element in block.findall(f'{w_ns}tr'):
                row_cells_text = []
                for cell_element in row_element.findall(f'{w_ns}tc'):
                    cell_text_parts = []
                    for paragraph_in_cell in cell_element.findall(f'{w_ns}p'):
                        cell_paragraph_text = extract_text_from_element(paragraph_in_cell, image_map)
                        cell_text_parts.append(cell_paragraph_text)
                    row_cells_text.append(" ".join(cell_text_parts).strip())
                plain_text_lines.append("\t".join(row_cells_text))

    return "\n".join(plain_text_lines)


# Main processing function
def process_docx_files():
    base_path = Path('files_clean/equipment')
    docx_files = list(base_path.rglob('*.docx'))
    
    doc_mapping = {}

    for index, filepath in enumerate(docx_files):
        filename_wo_extention = f"{str(filepath.name).split('.')[0]}"
        output_images_directory = f"files_preproc/{filename_wo_extention}"
        output_text_file = f"files_preproc/{filename_wo_extention}.txt"

        doc_mapping[f"D_{index}"] = filename_wo_extention

        document_obj, image_mapping = extract_and_map_images(filepath, output_images_directory, doc_prefix=f"D_{index}")
        
        # Use the enhanced function that handles hyperlinks
        processed_text = convert_docx_to_plain_text_with_markers(document_obj, image_mapping)
        
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(processed_text)

        print(f"Images extracted to: {output_images_directory}")
        print(f"Text with markers saved to: {output_text_file}")

    with open("files_preproc/doc_mapping.json", "w", encoding="utf-8") as f:
        json.dump(doc_mapping, f, indent=2, ensure_ascii=False)

def process_docx_file(file_path:str, doc_index:int):

    filepath = Path(file_path)

    filename_wo_extention = f"{str(filepath.name).split('.')[0]}"
    output_images_directory = f"files_preproc/output/{filename_wo_extention}"
    output_text_file = f"files_preproc/output/{filename_wo_extention}.txt"

    document_obj, image_mapping = extract_and_map_images(filepath, output_images_directory, doc_prefix=f"D_{doc_index}")
    
    # Use the enhanced function that handles hyperlinks
    processed_text = convert_docx_to_plain_text_with_markers(document_obj, image_mapping)
    processed_text = unidecode(processed_text)
    
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(processed_text)

    print(f"Images extracted to: {output_images_directory}")
    print(f"Text with markers saved to: {output_text_file}")


if __name__ == "__main__":
    # process_docx_files()
    process_docx_file(file_path='files_clean/equipment/MS112_um_eng.docx', doc_index=33)