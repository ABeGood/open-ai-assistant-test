import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from pathlib import Path
import threading
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
import json
from unidecode import unidecode

class DocxConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DOCX to Text Converter")
        self.root.geometry("600x300")
        
        # Get the directory where the executable/script is located
        if getattr(sys, 'frozen', False):
            # Running as exe
            self.exe_dir = os.path.dirname(sys.executable)
        else:
            # Running as script
            self.exe_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.default_output_dir = os.path.join(self.exe_dir, "outputs")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input file selection
        ttk.Label(main_frame, text="Select DOCX file:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.input_file_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_file_var, width=50)
        self.input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(input_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=1)
        
        # Output folder selection
        ttk.Label(main_frame, text="Output folder:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.output_folder_var = tk.StringVar(value=self.default_output_dir)
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_folder_var, width=50)
        self.output_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(output_frame, text="Browse", command=self.browse_output_folder).grid(row=0, column=1)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Convert DOCX", command=self.start_conversion)
        self.process_button.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status text
        self.status_text = tk.Text(main_frame, height=8, width=70)
        self.status_text.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for status text
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.grid(row=6, column=2, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        input_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select DOCX file",
            filetypes=[("Word documents", "*.docx"), ("All files", "*.*")]
        )
        if filename:
            self.input_file_var.set(filename)
            
    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_folder_var.set(folder)
            
    def log_message(self, message):
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
    def start_conversion(self):
        input_file = self.input_file_var.get().strip()
        output_folder = self.output_folder_var.get().strip()
        
        if not input_file:
            messagebox.showerror("Error", "Please select a DOCX file to convert.")
            return
            
        if not os.path.exists(input_file):
            messagebox.showerror("Error", "Selected input file does not exist.")
            return
            
        if not output_folder:
            output_folder = self.default_output_dir
            self.output_folder_var.set(output_folder)
            
        # Clear status text
        self.status_text.delete(1.0, tk.END)
        
        # Start conversion in a separate thread
        self.process_button.config(state='disabled')
        self.progress.start()
        
        thread = threading.Thread(target=self.convert_file, args=(input_file, output_folder))
        thread.daemon = True
        thread.start()
        
    def convert_file(self, input_file, output_folder):
        try:
            self.log_message(f"Starting conversion of: {input_file}")
            self.log_message(f"Output folder: {output_folder}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Process the file
            filepath = Path(input_file)
            filename_wo_extension = filepath.stem
            output_images_directory = os.path.join(output_folder, filename_wo_extension)
            output_text_file = os.path.join(output_folder, f"{filename_wo_extension}.txt")
            
            self.log_message("Extracting images and processing document...")
            
            document_obj, image_mapping = self.extract_and_map_images(filepath, output_images_directory)
            processed_text = self.convert_docx_to_plain_text_with_markers(document_obj, image_mapping)
            processed_text = unidecode(processed_text)
            
            with open(output_text_file, "w", encoding="utf-8") as f:
                f.write(processed_text)
            
            self.log_message(f"✓ Images extracted to: {output_images_directory}")
            self.log_message(f"✓ Text saved to: {output_text_file}")
            self.log_message("✓ Conversion completed successfully!")
            
            messagebox.showinfo("Success", "Conversion completed successfully!")
            
        except Exception as e:
            error_msg = f"Error during conversion: {str(e)}"
            self.log_message(f"✗ {error_msg}")
            messagebox.showerror("Error", error_msg)
        
        finally:
            self.progress.stop()
            self.process_button.config(state='normal')

    def extract_and_map_images(self, document_path, output_image_dir):
        document = Document(document_path)
        image_map = {}
        
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        image_count = 0
        for rel in document.part.rels:
            if "image" in document.part.rels[rel].reltype:
                image_part = document.part.rels[rel].target_part
                image_bytes = image_part.blob
                image_extension = image_part.content_type.split('/')[-1]
                
                new_filename = f"IMG_{image_count:03d}.{image_extension}"
                output_path = os.path.join(output_image_dir, new_filename)
                
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                
                image_map[rel] = new_filename
                image_count += 1
                
        return document, image_map

    def extract_text_from_element(self, element, image_map):
        text_parts = []
        
        w_ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        a_ns = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
        r_ns = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
        
        for t in element.findall(f'{w_ns}t'):
            if t.text:
                text_parts.append(t.text)
        
        for hyperlink in element.findall(f'{w_ns}hyperlink'):
            hyperlink_text = self.extract_text_from_element(hyperlink, image_map)
            text_parts.append(hyperlink_text)
        
        for run in element.findall(f'{w_ns}r'):
            for t in run.findall(f'{w_ns}t'):
                if t.text:
                    text_parts.append(t.text)
            
            for drawing in run.findall(f'{w_ns}drawing'):
                blip_elements = drawing.findall(f'.//{a_ns}blip')
                for blip in blip_elements:
                    r_id = blip.get(f'{r_ns}embed')
                    if r_id in image_map:
                        marker_name = image_map[r_id].split('.')[0]
                        text_parts.append(f"@{marker_name}")
                        break
        
        return "".join(text_parts)

    def convert_docx_to_plain_text_with_markers(self, document, image_map):
        plain_text_lines = []
        
        w_ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

        for block in document.element.body:
            if block.tag.endswith('p'):
                paragraph_text = self.extract_text_from_element(block, image_map)
                plain_text_lines.append(paragraph_text)

            elif block.tag.endswith('tbl'):
                previous_row_values = []
                for row_index, row_element in enumerate(block.findall(f'{w_ns}tr')):
                    row_cells_text = []
                    for cell_index, cell_element in enumerate(row_element.findall(f'{w_ns}tc')):
                        cell_text_parts = []
                        for paragraph_in_cell in cell_element.findall(f'{w_ns}p'):
                            cell_paragraph_text = self.extract_text_from_element(paragraph_in_cell, image_map)
                            cell_text_parts.append(cell_paragraph_text)
                        cell_text = " ".join(cell_text_parts).strip()
                        
                        if not cell_text and row_index > 0 and cell_index < len(previous_row_values):
                            cell_text = previous_row_values[cell_index]
                        
                        row_cells_text.append(cell_text)
                    
                    previous_row_values = row_cells_text[:]
                    plain_text_lines.append("\t".join(row_cells_text))

        return "\n".join(plain_text_lines)


def main():
    root = tk.Tk()
    app = DocxConverterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()