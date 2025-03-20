import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image
import PyPDF2

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    EasyOcrOptions,
    TesseractCliOcrOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem, TextItem, DoclingDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFIngestor:
    """
    A class for processing PDF documents with multiple OCR engines.
    
    This processor breaks PDFs into manageable batches, extracts text and structural
    information using Docling with various OCR engines, and saves the results in JSON format to disk.
    Images are also extracted and saved to disk in the process.
    """
    
    def __init__(self, config_path: str = "config/config.json", 
                 config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the PDF processor.
        
        Args:
            config_path: Path to the configuration JSON file
            config_override: Optional configuration dictionary to override loaded settings
        """
        # Load configuration from JSON file
        self.config = self._load_config(config_path)
        
        # Apply any overrides
        if config_override:
            self._update_config(config_override)
        
        # Setup output directory from config
        self.output_dir = Path(self.config.get("OUTPUT_FOLDER", "output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize converters for each OCR engine
        self.converters = self._create_converters()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            raise
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration with user-provided values.
        
        Args:
            config: Dictionary of configuration values to override loaded config
        """
        # Handle base level and DOCLING nested dictionary
        for key, value in config.items():
            if key == "DOCLING" and isinstance(value, dict) and "DOCLING" in self.config:
                # Update nested DOCLING dictionary
                for docling_key, docling_value in value.items():
                    if docling_key == "OCR_ENGINES" and isinstance(docling_value, dict) and "OCR_ENGINES" in self.config["DOCLING"]:
                        # Special handling for OCR_ENGINES to ensure nested updates
                        for engine_name, engine_config in docling_value.items():
                            if engine_name in self.config["DOCLING"]["OCR_ENGINES"]:
                                # Update existing engine config instead of replacing it
                                if isinstance(engine_config, dict) and isinstance(self.config["DOCLING"]["OCR_ENGINES"][engine_name], dict):
                                    self.config["DOCLING"]["OCR_ENGINES"][engine_name].update(engine_config)
                                else:
                                    self.config["DOCLING"]["OCR_ENGINES"][engine_name] = engine_config
                            else:
                                # Add new engine config
                                self.config["DOCLING"]["OCR_ENGINES"][engine_name] = engine_config
                    elif (isinstance(docling_value, dict) and 
                            docling_key in self.config["DOCLING"] and 
                            isinstance(self.config["DOCLING"][docling_key], dict)):
                        # Regular update for other nested dictionaries
                        self.config["DOCLING"][docling_key].update(docling_value)
                    else:
                        # Simple replacement for non-dictionary values
                        self.config["DOCLING"][docling_key] = docling_value
            else:
                # Update base level keys
                self.config[key] = value
    
    def _create_converters(self) -> Dict[str, DocumentConverter]:
        """
        Create document converters for all configured OCR engines.
        
        Returns:
            Dictionary mapping engine names to their respective DocumentConverter instances
        """
        converters = {}
        
        # Get the OCR engines dictionary from the config
        ocr_engines = self.config["DOCLING"]["OCR_ENGINES"]
        
        # Create converters for each enabled OCR engine
        for engine_name, engine_config in ocr_engines.items():
            # Skip disabled engines
            if not engine_config.get("ENABLED", True):
                logger.info(f"Skipping disabled OCR engine: {engine_name}")
                continue
                
            converters[engine_name] = self._create_converter(engine_name, engine_config)
            
        return converters
    
    def _create_converter(self, ocr_engine: str, engine_config: Dict[str, Any]) -> DocumentConverter:
        """
        Create a document converter with a specific OCR configuration.
        
        Args:
            ocr_engine: Name of the OCR engine to configure
            engine_config: Configuration for the specific OCR engine
            
        Returns:
            Configured DocumentConverter instance
        """
        pipeline_options = PdfPipelineOptions()
        
        # Configure visualization options
        pipeline_options.images_scale = self.config["DOCLING"]["IMAGES_SCALE"]
        
        # Configure accelerator options
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=self.config["DOCLING"]["THREADS"],
            device=self.config["DOCLING"]["DEVICE"]
        )
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=24,
            device=AcceleratorDevice.CUDA
        )
        
        # Configure OCR options based on engine
        if ocr_engine == "NOOCR":
            # No OCR configuration needed
            pipeline_options.do_ocr = False
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True
        elif ocr_engine == "RAPIDOCR":
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = RapidOcrOptions(
                lang=engine_config["LANG"],
                force_full_page_ocr=engine_config["FORCE_FULL_PAGE"],
                bitmap_area_threshold=engine_config["BITMAP_THRESHOLD"],
                text_score=engine_config["TEXT_SCORE"]
            )
        elif ocr_engine == "EASYOCR":
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = EasyOcrOptions(
                confidence_threshold=engine_config["CONFIDENCE"],
                bitmap_area_threshold=engine_config["BITMAP_THRESHOLD"],
                lang=engine_config["LANG"],
                recog_network=engine_config["RECOG_NETWORK"],
                force_full_page_ocr=engine_config["FORCE_FULL_PAGE"],
                download_enabled=engine_config["DOWNLOAD_ENABLED"]
            )
        elif ocr_engine == "TESSERACT":
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = TesseractCliOcrOptions(
                lang=engine_config["LANG"],
                force_full_page_ocr=engine_config["FORCE_FULL_PAGE"],
                bitmap_area_threshold=engine_config["BITMAP_THRESHOLD"],
                tesseract_cmd=engine_config["CMD"]
            )
        
        # Configure table extraction options
        table_config = self.config["DOCLING"]["TABLE"]
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode(table_config["MODE"])
        pipeline_options.table_structure_options.do_cell_matching = table_config["CELL_MATCHING"]
        
        # Create and return converter
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    
    def ingest_pdf(self, pdf_path: str) -> None:
        """
        Ingests and processes a PDF file in batches using all configured OCR engines.
        
        Args:
            pdf_path: Path to the PDF file to process
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Generate file ID with timestamp and hash
        timestamp = datetime.now().strftime(self.config["FILE_ID_FORMAT"])
        file_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]
        file_id = f"{timestamp}_{file_hash}"
        
        try:
            # Validate PDF and get page count
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"Processing PDF with {total_pages} pages")
            
            # Process in batches
            for start_page in range(0, total_pages, self.config["MAX_PAGES_PER_BATCH"]):
                end_page = min(start_page + self.config["MAX_PAGES_PER_BATCH"], total_pages)
                logger.info(f"Processing batch: pages {start_page + 1} to {end_page}")
                
                # Extract batch to temporary PDF
                temp_pdf = self.extract_batch(pdf_path, start_page, end_page)
                try:
                    # Process the batch with no-ocr first to extract structure and images
                    batch_data = self.process_batch(temp_pdf, file_id, start_page, total_pages)
                    
                    # Then process with each OCR engine and save results
                    self.process_batch_with_engines(temp_pdf, batch_data, file_id, start_page, total_pages)
                finally:
                    # Clean up temporary file
                    if temp_pdf.exists():
                        temp_pdf.unlink()
        
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
            raise
    
    def extract_batch(self, pdf_path: Path, start_page: int, end_page: int) -> Path:
        """
        Extract a batch of pages to a temporary PDF.
        
        Args:
            pdf_path: Path to the original PDF
            start_page: Start page index (0-based)
            end_page: End page index (exclusive)
            
        Returns:
            Path to the temporary PDF containing the extracted batch
        """
        logger.info(f"Extracting pages {start_page} to {end_page}")
        
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        os.close(temp_fd)
        temp_path = Path(temp_path)
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pdf_writer = PyPDF2.PdfWriter()
            
            # Add pages to the writer (PyPDF2 uses 0-based indexing)
            for page_num in range(start_page, min(end_page, len(pdf_reader.pages))):
                pdf_writer.add_page(pdf_reader.pages[page_num])
            
            # Save the temporary PDF
            with open(temp_path, 'wb') as output_file:
                pdf_writer.write(output_file)
        
        return temp_path
    
    def process_batch(self, temp_pdf: Path, file_id: str, start_page: int, total_pages: int) -> Dict[int, Dict]:
        """
        Process a batch of pages from the temporary PDF using no-OCR mode.
        This extracts document structure and saves images.
        
        Args:
            temp_pdf: Path to the temporary PDF containing the batch
            file_id: Unique identifier for the file
            start_page: Starting page number in original document
            total_pages: Total number of pages in original document
            
        Returns:
            Dictionary mapping page numbers to their initial batch data
        """
        logger.info(f"Processing batch with no-OCR (document structure extraction)")
        
        # Process with no-OCR engine
        engine_name = "NOOCR"
        
        # Make sure the NOOCR engine is available in the converters
        if engine_name not in self.converters:
            logger.error(f"No-OCR engine not found in converters")
            raise ValueError(f"No-OCR engine not available")
            
        result = self.converters[engine_name].convert(temp_pdf)
        doc = result.document
        
        # Store batch data for each page
        batch_data = {}
        
        # Process each page in the batch
        for local_idx, page in doc.pages.items():
            original_page = start_page + local_idx
            logger.info(f"Processing page {original_page} of {total_pages} (structure extraction)")
            
            # Save page image if available
            page_image_path = ""
            if page.image and page.image.pil_image:
                image_filename = f"{file_id}_page_{original_page}.png"
                page_image_path = str(self.output_dir / image_filename)
                page.image.pil_image.save(page_image_path)
                
                # Resize the page image if enabled
                if self.config.get("IMAGE", {}).get("RESIZE_IMAGE", False):
                    self._resize_image(page_image_path)
            
            # Extract image data from pictures
            image_data = self._extract_images(doc, local_idx, file_id, original_page)
            
            # Extract text content
            text_content = []
            table_data = []
            
            # Process items on this page
            for item, _ in doc.iterate_items(page_no=local_idx):
                if isinstance(item, TextItem):
                    text_content.append(item.text)
                elif isinstance(item, TableItem):
                    if item.data and item.data.grid:
                        table = []
                        for row in item.data.grid:
                            table_row = []
                            for cell in row:
                                cell_text = cell.text.strip()
                                table_row.append(cell_text)
                            table.append(table_row)
                        table_data.append(table)
            
            # Initialize content structure
            content = {
                "page_summary": "",
                "text_data": "",
                "page_description": "",
                "image_data": image_data,
                f"rawtext_{engine_name}": "\n".join(text_content),
                f"rawtable_{engine_name}": table_data,
            }
            
            # Create page data object
            page_data = {
                "file_id": file_id,
                "file_type": "pdf",
                "file_name": str(temp_pdf.name),
                "file_description": "",
                "page_number": original_page,
                "page_numbers": total_pages,
                "page_image": image_filename,
                "content": content
            }
            
            # Store in batch data
            batch_data[original_page] = page_data
            
            # Write initial JSON output
            self._save_page_data(page_data, file_id, original_page)
        
        return batch_data
    
    def process_batch_with_engines(self, temp_pdf: Path, batch_data: Dict[int, Dict], 
                                   file_id: str, start_page: int, total_pages: int) -> None:
        """
        Process a batch with all remaining OCR engines and update the JSON files.
        
        Args:
            temp_pdf: Path to the temporary PDF containing the batch
            batch_data: Initial batch data from no-OCR processing
            file_id: Unique identifier for the file
            start_page: Starting page number in original document
            total_pages: Total number of pages in original document
        """
        # Skip the no-OCR engine as it's already been processed
        for engine_name, converter in [(name, conv) for name, conv in self.converters.items() if name != "NOOCR"]:
            logger.info(f"Processing batch with {engine_name}")
            
            result = converter.convert(temp_pdf)
            doc = result.document
            
            # Process each page in the batch
            for local_idx, page in doc.pages.items():
                original_page = start_page + local_idx
                logger.info(f"Processing page {original_page} of {total_pages} with {engine_name}")
                
                if original_page not in batch_data:
                    logger.warning(f"Page {original_page} not found in batch data")
                    continue
                
                # Extract text content with this OCR engine
                text_content = []
                table_data = []
                
                # Process items on this page
                for item, _ in doc.iterate_items(page_no=local_idx):
                    if isinstance(item, TextItem):
                        text_content.append(item.text)
                    elif isinstance(item, TableItem):
                        if item.data and item.data.grid:
                            table = []
                            for row in item.data.grid:
                                table_row = []
                                for cell in row:
                                    cell_text = cell.text.strip() 
                                    table_row.append(cell_text)
                                table.append(table_row)
                            table_data.append(table)
                
                # Update the content for this engine
                batch_data[original_page]["content"][f"rawtext_{engine_name}"] = "\n".join(text_content)
                batch_data[original_page]["content"][f"rawtable_{engine_name}"] = table_data
                
                # Save updated data
                self._save_page_data(batch_data[original_page], file_id, original_page)

    def _extract_images(self, doc: DoclingDocument, local_idx: int, file_id: str, 
                    original_page: int) -> List[Dict[str, str]]:
        """
        Extract images from a document page and save them to disk.
        
        Args:
            doc: DoclingDocument containing the page
            local_idx: Local page index in the document
            file_id: Unique identifier for the file
            original_page: Original page number in the source document
            
        Returns:
            List of dictionaries containing image paths and metadata
        """
        image_data = []
        image_counter = 1  # Initialize a counter specifically for images
        
        for _, (item, _) in enumerate(doc.iterate_items(page_no=local_idx)):
            if isinstance(item, PictureItem) and item.image and item.image.pil_image:
                # Use the image-specific counter instead of the loop index
                image_filename = f"{file_id}_page_{original_page}_img_{image_counter}.png"
                image_path = self.output_dir / image_filename
                item.image.pil_image.save(image_path)
                
                # Resize the image if enabled
                if self.config.get("IMAGE", {}).get("RESIZE_IMAGE", False):
                    self._resize_image(str(image_path))
                
                # Create image metadata
                image_info = {
                    "image_filename": image_filename,
                    "image_description": ""
                }
                
                # Add any caption text if available
                # Use a safer approach to access captions
                try:
                    if hasattr(item, 'captions') and item.captions:
                        captions = []
                        for caption_ref in item.captions:
                            # Try different methods to access the caption text
                            try:
                                # First try direct access if caption_ref is the text itself
                                if isinstance(caption_ref, str):
                                    captions.append(caption_ref)
                                # Then try to access through doc methods if available
                                elif hasattr(doc, 'resolve_ref'):
                                    caption_item = doc.resolve_ref(caption_ref)
                                    if caption_item and hasattr(caption_item, 'text'):
                                        captions.append(caption_item.text)
                                # Try direct property access on caption_ref
                                elif hasattr(caption_ref, 'text'):
                                    captions.append(caption_ref.text)
                            except Exception as e:
                                logger.warning(f"Error accessing caption: {str(e)}")
                        
                        if captions:
                            image_info["image_description"] = " ".join(captions)
                except Exception as e:
                    logger.warning(f"Error processing captions: {str(e)}")
                
                image_data.append(image_info)
                image_counter += 1  # Increment the image counter only when an image is processed
        
        return image_data
    
    def _resize_image(self, image_path: str) -> bool:
        """
        Process an image to ensure it meets total pixel requirements while preserving aspect ratio.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            bool: True if processing was successful, False otherwise
        """
        image_config = self.config.get("IMAGE", {})
        min_resolution = image_config.get("MIN_RESOLUTION", 224 * 224)
        max_resolution = image_config.get("MAX_RESOLUTION", 1920 * 1080)
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                total_pixels = width * height
                aspect_ratio = width / height
                
                # Check minimum resolution
                if total_pixels < min_resolution:
                    # Calculate new dimensions maintaining aspect ratio
                    scale_factor = (min_resolution / total_pixels) ** 0.5
                    new_width = int(round(width * scale_factor))
                    new_height = int(round(height * scale_factor))
                
                # Check maximum resolution
                elif total_pixels > max_resolution:
                    # Calculate new dimensions maintaining aspect ratio
                    scale_factor = (max_resolution / total_pixels) ** 0.5
                    new_width = int(round(width * scale_factor))
                    new_height = int(round(height * scale_factor))
                
                else:
                    new_width, new_height = width, height
                
                if (new_width, new_height) != (width, height):
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    resized_img.save(image_path, quality=95)
                    logger.info(f"({image_path}) - Image resized to {new_width}x{new_height} ({new_width * new_height/1000000:.2f} megapixels)")
                else:
                    logger.debug(f"({image_path}) - Image already meets resolution requirements ({width}*{height}) ({total_pixels/1000000:.2f} megapixels)")
                return True
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return False
    
    def _save_page_data(self, page_data: Dict, file_id: str, page_number: int) -> None:
        """
        Save page data to a JSON file.
        
        Args:
            page_data: Dictionary containing page data
            file_id: Unique identifier for the file
            page_number: Page number
        """
        json_filename = f"{file_id}_page_{page_number}.json"
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved page data to {json_path}")


def main():
    """Main function to demonstrate the PDF processor usage."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process a PDF document using PDFIngestor.")
    parser.add_argument("pdf_path", help="Path to the PDF document to process")

    args = parser.parse_args()
    
    # Create an ingestor and run
    ingestor = PDFIngestor()
    ingestor.ingest_pdf(args.pdf_path)
    
    print(f"PDF ingestion completed successfully for: {args.pdf_path}")


if __name__ == "__main__":
    main()