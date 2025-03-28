import json
import logging
import os
import copy
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from llama_server_connector import LlamaServerConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChatProcessor:
    """
    A class for processing extracted PDF content using LLM models.
    
    This processor takes the output of PDFIngestor, with image descriptions from ImageDescriptor,
    and enriches it with LLM-generated text analysis of the OCR results.
    """
    
    def __init__(self, config_path: str = "config/config.json", 
                 config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the ChatProcessor.
        
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
        
        # Base prompt for OCR fixing
        self.base_ocr_prompt = """
Find above four different text extraction outputs from the same PDF page: three OCR outputs created with three different OCR engines, as well as the raw text of the page extracted without OCR (the NO_OCR_Input). Think about what the text may be talking about. Then, rewrite the text output to reflect the most likely original text/table, considering the document type and context; you may add additional text to it if an obvious omission is detected. You must do your best to preserve/enhance/fix the table structure, if table data is present.

Bear in mind that extracted text may be prone to errors, including, but not limited to: formatting, wrong characters, wrong paragraph ordering, mispelled or misplaced words, etc. Typically the NO OCR input is the most accurate for text, but it may lack important information that OCR engines can provide, especially when it comes to table data and formatting.

Your goal is to provide a Markdown version of the correct text, using these inputs as your base. You MUST output ONLY the rewritten text/table without any additional explanations."""

        # Base prompt for content summary
        self.base_summary_prompt = """
Read the description of a slide from a PDF file, created by a vision-enabled LLM, as well as the output of an OCR process on the same slide. You must aggregate ALL information contained in both of these outputs.

Use the page description as your main source of information, as it may contain information that is not present in the OCR output, such as, for instance, the interpretation of the slide itself and/or its components (slides, figures, tables etc). Use the OCR output to correct eventual errors or omissions, bearing in mind that a vision-enabled LLM is usually bad at reading text.

To be clear, your task is to produce a single output which will represent the totality of the information contained in the page. You must output ONLY the consolidated content of the page, without any additional explanations. Preserve all information contained in both inputs at all costs. Prefer markdown formatting on your response."""
    
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
        for key, value in config.items():
            if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def _load_models_config(self, models_path: str = "config/models.json") -> Dict[str, Any]:
        """
        Load models configuration from the models.json file.
        
        Args:
            models_path: Path to the models configuration JSON file
            
        Returns:
            Models configuration dictionary
        """
        try:
            with open(models_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading models config from {models_path}: {str(e)}")
            raise
    
    def _build_chat_model_config(self, chat_model_override: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Build a chat model configuration by combining models.json and config.json settings.
        
        Args:
            chat_model_override: Optional dictionary to override the DEFAULT_CHAT_MODEL settings
            
        Returns:
            Tuple of (model_key, parameter_overrides)
        """
        # Load the base models configuration
        base_models_config = self._load_models_config()
        
        # Get the chat model settings, with override if provided
        chat_model_config = chat_model_override if chat_model_override else self.config.get("DEFAULT_CHAT_MODEL", {})
        model_name = chat_model_config.get("MODEL_NAME", "DEEPSEEK-R1-QWEN-14B")
        
        # Ensure the target model exists in the configuration
        if model_name not in base_models_config.get("MODELS", {}):
            logger.warning(f"Model {model_name} not found in models.json.")
            if "DEEPSEEK-R1-QWEN-14B" in base_models_config.get("MODELS", {}):
                logger.info("Falling back to DEEPSEEK-R1-QWEN-14B model")
                model_name = "DEEPSEEK-R1-QWEN-14B"
            else:
                # Use the first available model as fallback
                model_name = next(iter(base_models_config.get("MODELS", {}).keys()))
                logger.info(f"Falling back to {model_name} model")
        
        # Create parameter overrides dictionary (exclude MODEL_NAME)
        param_overrides = {k: v for k, v in chat_model_config.items() if k != "MODEL_NAME"}
        
        # Return the model key and parameter overrides
        return model_name, param_overrides
    
    def dict_to_markdown(self, data, level=0):
        """
        Convert dictionary to Markdown format with proper indentation
        
        Args:
            data: Dictionary to convert
            level: Current indentation level
        
        Returns:
            String in Markdown format
        """
        markdown = ""
        indent = "  " * level
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    markdown += f"{indent}## {key}\n"
                    markdown += self.dict_to_markdown(value, level + 1)
                else:
                    markdown += f"{indent}**{key}:** {value}\n"
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    markdown += f"{indent}- \n{self.dict_to_markdown(item, level + 1)}"
                else:
                    markdown += f"{indent}- {item}\n"
        else:
            markdown += f"{indent}{data}\n"
        
        return markdown
    
    def create_ocr_fix_prompt(self, page_data: Dict[str, Any]) -> str:
        """
        Create a prompt for fixing OCR issues based on multiple OCR outputs.
        
        Args:
            page_data: Dictionary containing page data with multiple OCR outputs
            
        Returns:
            Prompt string for the LLM
        """
        # Extract context and OCR dictionaries
        context_dict = copy.deepcopy(page_data)
        
        # Extract OCR dictionaries for each engine
        ocr_dict1 = {
            'text_data': page_data['content'].get('rawtext_RAPIDOCR', ''),
            'table_data': page_data['content'].get('rawtable_RAPIDOCR', [])
        }
        
        ocr_dict2 = {
            'text_data': page_data['content'].get('rawtext_EASYOCR', ''),
            'table_data': page_data['content'].get('rawtable_EASYOCR', [])
        }
        
        ocr_dict3 = {
            'text_data': page_data['content'].get('rawtext_TESSERACT', ''),
            'table_data': page_data['content'].get('rawtable_TESSERACT', [])
        }
        
        ocr_dict4 = {
            'text_data': page_data['content'].get('rawtext_NOOCR', ''),
            'table_data': page_data['content'].get('rawtable_NOOCR', [])
        }
        
        # Mask the OCR content in the context dictionary
        if 'content' in context_dict:
            content = context_dict['content']
            for key in [
                'rawtext_RAPIDOCR', 'rawtable_RAPIDOCR',
                'rawtext_EASYOCR', 'rawtable_EASYOCR',
                'rawtext_TESSERACT', 'rawtable_TESSERACT',
                'rawtext_NOOCR', 'rawtable_NOOCR',
                'text_data', 'table_data'
            ]:
                if key in content:
                    del content[key]
        
        # Convert dictionaries to markdown
        context_content = self.dict_to_markdown(context_dict)
        ocr_content1 = self.dict_to_markdown(ocr_dict1)
        ocr_content2 = self.dict_to_markdown(ocr_dict2)
        ocr_content3 = self.dict_to_markdown(ocr_dict3)
        ocr_content4 = self.dict_to_markdown(ocr_dict4)
        
        # Create the final prompt template
        prompt = f"""<Page_Context>
{context_content}</Page_Context>

<OCR_Input1>
{ocr_content1}</OCR_Input1>

<OCR_Input2>
{ocr_content2}</OCR_Input2>

<OCR_Input3>
{ocr_content3}</OCR_Input3>

<NO_OCR_Input>
{ocr_content4}</NO_OCR_Input>

{self.base_ocr_prompt}"""
        
        return prompt

    def create_content_summary_prompt(self, page_data: Dict[str, Any]) -> str:
        """
        Create a summary of the content on the page based on the fixed OCR output and the description of the image.
        
        Args:
            page_data: Dictionary containing page data
            
        Returns:
            Prompt string for the LLM
        """

        # Extract the text data and the page description
        text_data = page_data['content'].get('text_data', ''),
        page_description = page_data['content'].get('page_description', '')
        
        # Create the final prompt template
        prompt = f"""<Page_Description>
{page_description}</Page_Description>

<Page_OCR_Data>
{text_data}</Page_OCR_Data>

{self.base_summary_prompt}"""
        
        return prompt
    
    def route_task(self, task: str, page_data: Dict[str, Any]) -> str:
        """
        Route the task to the appropriate prompt creation method.
        
        Args:
            task: Task type, e.g., "ocr_fix"
            page_data: Dictionary containing page data
            
        Returns:
            Prompt string for the LLM
        """
        if task == "ocr_fix":
            return self.create_ocr_fix_prompt(page_data)
        elif task == "summary":
            return self.create_content_summary_prompt(page_data)
        # Add more tasks here if needed in the future
        else:
            logger.warning(f"Unknown task '{task}', defaulting to OCR fix task")
            return self.create_ocr_fix_prompt(page_data)
    
    def chat_with_data(self, output_dir: Optional[str] = None, 
                        llm_task: str = "ocr_fix",
                        chat_model_override: Optional[Dict[str, Any]] = None) -> None:
        """
        Process JSON files using LLM to enhance OCR output and store results.
        
        Args:
            output_dir: Optional path to the output directory (overrides the one in config)
            llm_task: The LLM task to perform (default is "ocr_fix")
            chat_model_override: Optional dictionary to override the DEFAULT_CHAT_MODEL settings
        """
        # Use provided output directory or default from config
        output_path = Path(output_dir) if output_dir else self.output_dir
        
        if not output_path.exists():
            logger.error(f"Output directory {output_path} does not exist")
            raise FileNotFoundError(f"Output directory {output_path} does not exist")
        
        # Get the model key and parameter overrides
        model_key, param_overrides = self._build_chat_model_config(chat_model_override)
        logger.info(f"Using model: {model_key} with overrides: {param_overrides}")
        
        try:
            logger.info(f"Looking for model key: '{model_key}' in models.json")
            # Print all available model keys for comparison
            with open("config/models.json", 'r') as f:
                models_config = json.load(f)
                logger.info(f"Available models: {list(models_config.get('MODELS', {}).keys())}")
            
            # Initialize LlamaServerConnector with default config path, our model key, and parameter overrides
            connector = LlamaServerConnector(
                config_path="config/models.json",
                model_key=model_key,
                param_overrides=param_overrides
            )
            
            # Get all JSON files in the output directory
            json_files = list(output_path.glob("*.json"))
            total_files = len(json_files)
            logger.info(f"Found {total_files} JSON files to process")
            
            # Start timing for progress tracking
            start_time = time.time()
            
            # Process each JSON file
            for index, json_file in enumerate(json_files):
                try:
                    # Calculate progress
                    current_file = index + 1
                    
                    # Timing calculations
                    elapsed_time = time.time() - start_time
                    average_time_per_file = elapsed_time / current_file if current_file > 0 else 0
                    remaining_files = total_files - current_file
                    estimated_time_remaining = average_time_per_file * remaining_files
                    estimated_completion_time = time.strftime("%H:%M:%S", time.localtime(time.time() + estimated_time_remaining))
                    
                    # Log progress
                    logger.info(f"Processing file {current_file}/{total_files} ({(current_file/total_files)*100:.1f}%) - "
                                f"Elapsed: {str(timedelta(seconds=int(elapsed_time)))} - "
                                f"Est. completion at: {estimated_completion_time}")
                    
                    # Load the JSON data
                    with open(json_file, 'r', encoding='utf-8') as f:
                        page_data = json.load(f)
                    
                    # Extract file_id and page_number for logging
                    file_id = page_data.get("file_id", "unknown")
                    page_number = page_data.get("page_number", "unknown")
                    logger.info(f"Processing {file_id} page {page_number}")
                    
                    # Create prompt based on the task
                    prompt = self.route_task(llm_task, page_data)
                    
                    # Get response from the LLM
                    logger.info(f"Sending prompt to LLM for {file_id} page {page_number}")
                    response = connector.get_response(prompt)

                    # Remove everything from the beginning up to and including "</think>" if it exists
                    if "</think>" in response:
                        response = response.split("</think>", 1)[1]
                    
                    if response:
                        # Update text_data or page_summary in the JSON data
                        if llm_task == "ocr_fix":
                            page_data["content"]["text_data"] = response
                            logger.info(f"Added LLM-fixed text for {file_id} page {page_number}")
                        elif llm_task == "summary":
                            page_data["content"]["page_summary"] = response
                            logger.info(f"Added LLM summary for {file_id} page {page_number}")                            
                        
                        # Save updated JSON data back to file
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(page_data, f, ensure_ascii=False, indent=2)
                    else:
                        logger.warning(f"Failed to get LLM response for {file_id} page {page_number}")
                
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {str(e)}", exc_info=True)
            
            # Log completion time statistics
            total_elapsed_time = time.time() - start_time
            logger.info(f"Processing completed. Total time: {str(timedelta(seconds=int(total_elapsed_time)))}, "
                        f"Average per file: {str(timedelta(seconds=int(total_elapsed_time/total_files)))}")
            
            # After processing all files, kill the LlamaServerConnector
            connector.kill_server()
            logger.info("LLM processing completed and server shut down")
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    """Main function to demonstrate the chat processing."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process text data using ChatProcessor.")
    parser.add_argument("--output", "-o", help="Output directory containing JSON files", default="output")
    parser.add_argument("--task", "-t", help="Task to perform (default: ocr_fix)", default="ocr_fix")
    
    args = parser.parse_args()
    
    # Create a ChatProcessor and run
    processor = ChatProcessor()
    processor.chat_with_data(output_dir=args.output, llm_task=args.task)
    
    print(f"Chat processing completed successfully for directory: {args.output}")