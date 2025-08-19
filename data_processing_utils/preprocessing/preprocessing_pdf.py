import fitz  # PyMuPDF
import os
from pathlib import Path
import json
import io
from PIL import Image

def extract_and_map_images(pdf_path, output_image_dir, doc_prefix=""):
    """
    Extract images from PDF and create mapping similar to DOCX version
    """
    pdf_document = fitz.open(pdf_path)
    image_map = {}  # Maps (page_num, img_index) to new filename
    
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    
    global_image_count = 0
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            # Get image data
            xref = img[0]
            pix = fitz.Pixmap(pdf_document, xref)
            
            # Convert CMYK to RGB if necessary
            if pix.n - pix.alpha < 4:  # GRAY or RGB
                img_data = pix.tobytes("png")
            else:  # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                img_data = pix1.tobytes("png")
                pix1 = None
            
            pix = None
            
            # Create filename
            new_filename = f"{doc_prefix}_IMG_{global_image_count:03d}.png"
            output_path = os.path.join(output_image_dir, new_filename)
            
            # Save image
            with open(output_path, "wb") as f:
                f.write(img_data)
            
            # Store mapping: (page_num, img_index) -> filename
            image_map[(page_num, img_index)] = new_filename
            global_image_count += 1
    
    pdf_document.close()
    return image_map

def get_image_positions(pdf_path):
    """
    Get positions of images in text to create proper markers
    """
    pdf_document = fitz.open(pdf_path)
    image_positions = {}  # Maps (page_num, img_index) to text position info
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Get text blocks with their positions
        text_dict = page.get_text("dict")
        image_list = page.get_images()
        
        # Get image rectangles
        for img_index, img in enumerate(image_list):
            # Get image rectangle
            img_rects = page.get_image_rects(img[0])
            if img_rects:
                img_rect = img_rects[0]  # Take first occurrence
                
                # Find the closest text block to determine insertion point
                blocks = text_dict["blocks"]
                best_block_idx = 0
                min_distance = float('inf')
                
                for block_idx, block in enumerate(blocks):
                    if "lines" in block:  # Text block
                        block_rect = fitz.Rect(block["bbox"])
                        # Calculate distance between image and text block
                        distance = abs(img_rect.y1 - block_rect.y0)
                        if distance < min_distance:
                            min_distance = distance
                            best_block_idx = block_idx
                
                image_positions[(page_num, img_index)] = {
                    'page': page_num,
                    'block_index': best_block_idx,
                    'rect': img_rect
                }
    
    pdf_document.close()
    return image_positions

def convert_pdf_to_plain_text_with_markers(pdf_path, image_map):
    """
    Extract text from PDF and insert image markers at appropriate positions
    """
    pdf_document = fitz.open(pdf_path)
    image_positions = get_image_positions(pdf_path)
    plain_text_lines = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Get text in blocks to maintain structure
        text_dict = page.get_text("dict")
        blocks = text_dict["blocks"]
        
        # Get images for this page
        page_images = [(k, v) for k, v in image_positions.items() if k[0] == page_num]
        
        for block_idx, block in enumerate(blocks):
            if "lines" in block:  # Text block
                block_text_parts = []
                
                # Extract text from lines
                for line in block["lines"]:
                    line_text_parts = []
                    for span in line["spans"]:
                        if span["text"].strip():
                            line_text_parts.append(span["text"])
                    
                    if line_text_parts:
                        block_text_parts.append("".join(line_text_parts))
                
                # Add block text
                if block_text_parts:
                    block_text = "\n".join(block_text_parts)
                    plain_text_lines.append(block_text)
                
                # Check if any images should be inserted after this block
                for (page, img_idx), img_filename in image_map.items():
                    if page == page_num:
                        pos_info = image_positions.get((page, img_idx))
                        if pos_info and pos_info['block_index'] == block_idx:
                            marker_name = img_filename.split('.')[0]
                            plain_text_lines.append(f"@{marker_name}")
    
    pdf_document.close()
    return "\n".join(plain_text_lines)

def extract_tables_from_pdf(pdf_path):
    """
    Extract tables from PDF if any (basic implementation)
    """
    try:
        import tabula
        # Extract tables using tabula-py
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        table_text = []
        
        for i, table in enumerate(tables):
            if not table.empty:
                # Convert table to tab-separated text
                table_lines = []
                for _, row in table.iterrows():
                    row_text = "\t".join([str(cell) if pd.notna(cell) else "" for cell in row])
                    table_lines.append(row_text)
                table_text.extend(table_lines)
        
        return "\n".join(table_text)
    except ImportError:
        # Fallback: use basic text extraction for tables
        return ""

def process_pdf_files():
    """
    Main processing function for PDF files
    """
    base_path = Path('files_clean/cables')
    pdf_files = list(base_path.rglob('*.pdf'))
    
    doc_mapping = {}
    
    for index, filepath in enumerate(pdf_files):
        filename_wo_extension = f"{str(filepath.name).split('.')[0]}"
        output_images_directory = f"files_preproc/{filename_wo_extension}"
        output_text_file = f"files_preproc/{filename_wo_extension}.txt"
        
        doc_mapping[f"D_{index}"] = filename_wo_extension
        
        # Extract images and create mapping
        image_mapping = extract_and_map_images(filepath, output_images_directory, doc_prefix=f"D_{index}")
        
        # Extract text with image markers
        processed_text = convert_pdf_to_plain_text_with_markers(filepath, image_mapping)
        
        # Save processed text
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(processed_text)
        
        print(f"Images extracted to: {output_images_directory}")
        print(f"Text with markers saved to: {output_text_file}")
    
    # Save document mapping
    with open("files_preproc/doc_mapping.json", "w", encoding="utf-8") as f:
        json.dump(doc_mapping, f, indent=2, ensure_ascii=False)

# Alternative simpler version if you prefer character-level positioning
def convert_pdf_simple_markers(pdf_path, image_map):
    """
    Simpler version: insert image markers at the end of each page that contains images
    """
    pdf_document = fitz.open(pdf_path)
    plain_text_lines = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Get page text
        page_text = page.get_text()
        if page_text.strip():
            plain_text_lines.append(page_text.strip())
        
        # Add image markers for this page
        page_images = [(k, v) for k, v in image_map.items() if k[0] == page_num]
        for (page, img_idx), img_filename in page_images:
            marker_name = img_filename.split('.')[0]
            plain_text_lines.append(f"@{marker_name}")
    
    pdf_document.close()
    return "\n".join(plain_text_lines)

if __name__ == "__main__":
    # Install required packages first:
    # pip install PyMuPDF pillow
    # Optional for table extraction: pip install tabula-py
    
    process_pdf_files()