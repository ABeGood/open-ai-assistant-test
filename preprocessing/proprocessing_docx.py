from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
import os
from pathlib import Path
import json

def extract_images(doc_path, output_dir, doc_id_prefix="DOC"):
    document = Document(doc_path)
    extracted_images_info = []
    image_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for r in document.part.rels:
        if document.part.rels[r].reltype == RT.IMAGE:
            image_part = document.part.rels[r].target_part
            image_bytes = image_part.blob
            image_extension = image_part.content_type.split('/')[-1]
            
            # Generate a unique and atomic name
            image_filename = f"{doc_id_prefix}_{image_count:03d}.{image_extension}"
            output_path = os.path.join(output_dir, image_filename)
            
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            
            extracted_images_info.append({
                "original_rid": r, # Relationship ID, useful for linking back
                "new_filename": image_filename,
                "output_path": output_path
            })
            image_count += 1
            
    return extracted_images_info


def get_plain_text(document, image_markers):
    full_text_lines = []
    
    # Track the images that have been "marked" to avoid duplicates if an image appears
    # in multiple places (unlikely but good for robustness).
    marked_rids = set() 

    for paragraph in document.paragraphs:
        paragraph_text = ""
        # Iterate through runs to detect images more precisely
        for run in paragraph.runs:
            if hasattr(run, 'element') and 'inline' in str(run.element.xml):
                # This is a very rough way to check for inline shapes.
                # A more robust check might involve parsing the XML or checking run.element.findall
                # for specific image-related tags.
                # However, for simplicity and to put the marker at the image's original *place*,
                # we'll iterate through the document's relationships.
                pass # Images will be handled when iterating doc.inline_shapes or doc.tables
            
            # Simple text extraction for now
            paragraph_text += run.text
        
        # Now, check for images within this paragraph
        # This part is tricky: python-docx doesn't directly tell you *where* in the text
        # a specific InlineShape (image) appears. It's usually associated with a 'run'.
        # The best approach for exact placement is to process the XML directly,
        # or use a placeholder in the DOCX itself that you replace.
        
        # A simpler approach (less precise for multi-image paragraphs):
        # If the paragraph *contains* an inline image, add its marker.
        # This is where the 'original_rid' from extract_images comes in handy.
        
        # For simplicity in this example, let's assume we'll just insert markers
        # at a paragraph level if any image is associated with it.
        # A more advanced solution would involve iterating through the XML of runs within a paragraph
        # and checking for <w:drawing> elements that contain images.

        full_text_lines.append(paragraph_text)
    
    for table in document.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                cell_text = ""
                for paragraph in cell.paragraphs:
                    cell_text += paragraph.text + " " # Add space for cell separation
                row_text.append(cell_text.strip())
            full_text_lines.append("\t".join(row_text)) # Tab-separate table cells
            
    return "\n".join(full_text_lines)


# Function to extract images and store their relationship IDs (rId)
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

# Function to convert text with markers
def convert_docx_to_plain_text_with_markers(document, image_map):
    plain_text_lines = []

    for block in document.element.body: # Iterate through the top-level XML elements (paragraphs, tables)
        if block.tag.endswith('p'): # It's a paragraph
            paragraph_text_parts = []
            for run_element in block.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r'):
                # Extract text from text runs
                t_elements = run_element.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
                for t in t_elements:
                    paragraph_text_parts.append(t.text)
                
                # Check for drawing elements (images) within the run
                drawing_elements = run_element.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing')
                for drawing in drawing_elements:
                    # Look for relationship ID in inline/anchor properties
                    # This path can be complex, and might vary slightly.
                    # Common path for inline images: <wp:inline><a:graphic><a:graphicData><pic:pic><pic:blipFill><a:blip r:embed="rIdX"/>
                    # For a full reliable parse, you might need to explore the XML structure of your specific DOCX.
                    
                    # A more generic way to find rId: look for <a:blip> with r:embed attribute
                    blip_elements = drawing.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
                    for blip in blip_elements:
                        r_id = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                        if r_id in image_map:
                            marker_name = image_map[r_id].split('.')[0]
                            paragraph_text_parts.append(f"@{marker_name}") # Insert the marker
                            break # Assume one image per blip for simplicity
            
            plain_text_lines.append("".join(paragraph_text_parts))

        elif block.tag.endswith('tbl'): # It's a table
            for row_element in block.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr'):
                row_cells_text = []
                for cell_element in row_element.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc'):
                    cell_text_parts = []
                    for paragraph_in_cell in cell_element.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                        for run_in_cell in paragraph_in_cell.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r'):
                            t_elements = run_in_cell.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
                            for t in t_elements:
                                cell_text_parts.append(t.text)
                            
                            drawing_elements = run_in_cell.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing')
                            for drawing in drawing_elements:
                                blip_elements = drawing.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
                                for blip in blip_elements:
                                    r_id = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                                    if r_id in image_map:
                                        marker_name = image_map[r_id].split('.')[0]
                                        cell_text_parts.append(f"@{marker_name}")
                                        break
                    row_cells_text.append("".join(cell_text_parts).strip())
                plain_text_lines.append("\t".join(row_cells_text)) # Use tabs to separate cells

    return "\n".join(plain_text_lines)



base_path = Path('files_clean/equipment')
docx_files = list(base_path.rglob('*.docx'))


doc_mapping = {}

for index, filepath in enumerate(docx_files):
    filename_wo_extention = f"{str(filepath.name).split('.')[0]}"
    output_images_directory = f"files_preproc/{filename_wo_extention}"
    output_text_file = f"files_preproc/{filename_wo_extention}.txt"

    doc_mapping[f"D_{index}"] = filename_wo_extention

    document_obj, image_mapping = extract_and_map_images(filepath, output_images_directory, doc_prefix=f"D_{index}")
    processed_text = convert_docx_to_plain_text_with_markers(document_obj, image_mapping)

    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(processed_text)


    print(f"Images extracted to: {output_images_directory}")
    print(f"Text with markers saved to: {output_text_file}")

with open("files_preproc/doc_mapping.json", "w", encoding="utf-8") as f:
    json.dump(doc_mapping, f, indent=2, ensure_ascii=False)
