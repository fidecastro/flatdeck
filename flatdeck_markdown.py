#!/usr/bin/env python
import os
import argparse
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_markdown_file(output_dir: str, pdf_name: str, output_type: str) -> None:
    """
    Build a markdown file with all page content in order.
    
    Args:
        output_dir (str): Directory containing processed JSON files
        pdf_name (str): Name of the original PDF file (for output filename)
    """
    output_path = Path(output_dir)
    
    # Get all JSON files in the output directory
    json_files = list(output_path.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {output_dir}")
        return
    
    # Extract file_id and page mapping
    file_data = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
                
            file_id = page_data.get("file_id", "unknown")
            page_number = page_data.get("page_number", 0)
            
            # Group by file_id
            if file_id not in file_data:
                file_data[file_id] = {}
                
            file_data[file_id][page_number] = page_data
        except Exception as e:
            logger.error(f"Error reading {json_file}: {str(e)}")
    
    # Process each file_id
    for file_id, pages in file_data.items():
        # Sort pages by page number
        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        
        # Create markdown content
        markdown_content = f"# {pdf_name}\n\n"
        
        for page in sorted_pages:
            page_number = page.get("page_number", "unknown")
            content = page.get("content", {})
            
            # Add strong page separator with page number
            markdown_content += f"# ======== PAGE {page_number} ========\n\n"
            
            # Add page summary if available
            page_summary = content.get("page_summary", "")
            if page_summary:
                markdown_content += f"{page_summary}\n\n"
                if output_type != "summary":
                    markdown_content += "- - - - - - - - - - - - - - - - - - - -\n\n"
            
            if output_type != "summary":
                # Add page description if available
                page_description = content.get("page_description", "")
                if page_description:
                    markdown_content += "### 📷 PAGE VISUAL DESCRIPTION\n\n"
                    markdown_content += f"{page_description}\n\n"
                    markdown_content += "- - - - - - - - - - - - - - - - - - - -\n\n"
                
                # Add text content if available
                text_data = content.get("text_data", "")
                if text_data:
                    markdown_content += "### 📝 PAGE CONTENT\n\n"
                    markdown_content += f"{text_data}\n\n"
                
                # Add only images that have descriptions
                image_data = content.get("image_data", [])
                for img in image_data:
                    if img.get("image_description"):
                        markdown_content += "- - - - - - - - - - - - - - - - - - - -\n\n"
                        markdown_content += f"### 🖼️ EMBEDDED IMAGE\n\n"
                        markdown_content += f"{img.get('image_description')}\n\n"
            
            # Add strong separator between pages
            markdown_content += "\n\n" # "# ====================\n\n"
        
        # Write markdown file
        markdown_file = output_path / f"{pdf_name}_{file_id}.md"
        try:
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.info(f"Markdown file created: {markdown_file}")
        except Exception as e:
            logger.error(f"Error writing markdown file: {str(e)}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate markdown from processed JSON files.")
    parser.add_argument("pdf_path", help="Path to the original PDF document (used for naming)")
    parser.add_argument("--output", "-o", help="Output directory containing JSON files", default="output")
    parser.add_argument("--output_type", "-t", help="Output type (summary or all)", default="summary")
    
    args = parser.parse_args()
    
    # Extract the PDF name without extension
    pdf_name = Path(args.pdf_path).stem
    
    # Build the markdown file
    logger.info(f"Building markdown file for {pdf_name}...")
    build_markdown_file(output_dir=args.output, pdf_name=pdf_name, output_type=args.output_type)
    
    logger.info("Markdown generation completed successfully!")


if __name__ == "__main__":
    main()