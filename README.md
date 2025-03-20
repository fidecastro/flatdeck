# FlatDeck

FlatDeck is an experimental PDF processing toolkit that extracts, analyzes, and enhances document content using local LLM models. It integrates multiple OCR engines with vision-capable AI models to provide comprehensive document understanding.

FlatDeck's goal is to flatten a PDF into a text file in markdown format that can be easily digested by LLMs. While any PDF can be used as input, it shines with visually rich documents -- it was designed with slide decks in mind.

## Features

- **Multiple OCR Engine Integration**: Combines RapidOCR, EasyOCR, and Tesseract outputs to improve text extraction accuracy
- **Vision Model Analysis**: Utilizes vision-capable LLMs (Gemma3, Qwen2-VL) to process images and diagrams
- **Text Enhancement**: Applies LLM processing to correct OCR errors and refine document content
- **Document Batching**: Processes large PDFs in configurable page batches
- **Configurable Pipeline**: Adjustable parameters via JSON configuration files

## Requirements

- Python 3.12
- llama.cpp compiled and in PATH (with gemma3 and qwen2vl variants for vision models)
- Docling library for document structure extraction
- Local GGUF model files as specified in config/models.json

## Installation (WIP)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flatdeck.git
   cd flatdeck
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place required model files in the `models/` directory according to config/models.json specifications

## Usage

Process a PDF document:
```bash
./flatdeck.sh document.pdf
```

## Processing Pipeline

FlatDeck operates in sequential stages:

1. **PDF Ingestion** (`flatdeck_pdf_ingestor.py`): Extracts document structure, page images, and embedded visual elements
2. **Image Description** (`flatdeck_image_descriptor.py`): Processes visual content using vision LLMs
3. **Text Enhancement** (`flatdeck_chat_processor.py`): Fixes the OCR outputs using LLM processing and aggregates all data into a summary containing all the data of the page

## Configuration

The system uses two primary configuration files:

- `config/config.json`: Core application parameters
- `config/models.json`: LLM model specifications and parameters

### Supported Models

The system was built on top of llama-cpp-connector[https://github.com/fidecastro/llama-cpp-connector/tree/main] to connect with GGUF models. All models supported by that project should be useful here:

**Vision Models:**
- GEMMA3 (12B and 27B variants)
- Qwen2-VL (7B)

**Text Processing Models:**
- DEEPSEEK-R1-QWEN-14B

## Output

FlatDeck generates output in the `output/` directory consisting of:

- JSON files containing comprehensive document data
- Extracted images with their LLM-generated descriptions
- Page images of the document

The final result is a markdown representation of each document page, including:
- Extracted and corrected text content
- Tabular data with preserved structure
- Visual elements with descriptive text
- Contextual relationships between document components

This complete digital representation preserves both the semantic content and the visual structure of the original document.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
