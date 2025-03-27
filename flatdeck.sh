#!/bin/bash
# FlatDeck - Shell script to process PDF documents through multiple steps

# Show usage if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 /path/to/document.pdf"
    exit 1
fi

# Get absolute path to the PDF
PDF_PATH=$(realpath "$1")
PDF_NAME=$(basename "$PDF_PATH" .pdf)


echo "==========================================="
echo "FlatDeck PDF Processing Pipeline"
echo "==========================================="
echo "PDF: $PDF_PATH"
echo "==========================================="

# Step 1: PDF Ingestion
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting PDF ingestion..."
python flatdeck_pdf_ingestor.py "$PDF_PATH"
if [ $? -ne 0 ]; then
    echo "ERROR: PDF ingestion failed!"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PDF ingestion completed."

# Allow some time for GPU resources to be released
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for GPU resources to be released..."
sleep 2

# Step 2: Image Description
echo "-------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting image description..."
python flatdeck_image_descriptor.py
if [ $? -ne 0 ]; then
    echo "WARNING: Image description failed. Continuing with processing..."
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Image description completed."

# Allow some time for GPU resources to be released
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for GPU resources to be released..."
sleep 2

# Step 3: OCR Text Processing
echo "-------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting OCR text processing..."
python flatdeck_chat_processor.py --task ocr_fix
if [ $? -ne 0 ]; then
    echo "WARNING: OCR text processing failed. Continuing with markdown generation..."
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] OCR text processing completed."

# Allow some time for GPU resources to be released
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for GPU resources to be released..."
sleep 2

# Step 4: Page Summary Processing
echo "-------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting page summary creation..."
python flatdeck_chat_processor.py --task summary
if [ $? -ne 0 ]; then
    echo "WARNING: Page summary processing failed. Continuing with markdown generation..."
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] OCR text processing completed."

# Step 5: Markdown Generation
echo "-------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating markdown output..."
python flatdeck_markdown.py "$PDF_PATH" --output_type summary
if [ $? -ne 0 ]; then
    echo "ERROR: Markdown generation failed!"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Markdown generation completed."

# Show output location
MARKDOWN_FILE="$OUTPUT_DIR/${PDF_NAME}_*.md"
echo "==========================================="
echo "Processing completed successfully!"
echo "Output markdown file(s): $MARKDOWN_FILE"
echo "==========================================="