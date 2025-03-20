import json
import logging
import os
import copy
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
            logger.info(f"Found {len(json_files)} JSON files to process")
            
            # Process each JSON file
            for json_file in json_files:
                try:
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
                    
                    if response:
                        # Update text_data in the JSON data
                        page_data["content"]["text_data"] = response
                        logger.info(f"Added LLM-processed text for {file_id} page {page_number}")
                        
                        # Save updated JSON data back to file
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(page_data, f, ensure_ascii=False, indent=2)
                    else:
                        logger.warning(f"Failed to get LLM response for {file_id} page {page_number}")
                
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {str(e)}", exc_info=True)
            
            # After processing all files, kill the LlamaServerConnector
            connector.kill_server()
            logger.info("LLM processing completed and server shut down")
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}", exc_info=True)
            raise


# Example usage
if __name__ == "__main__":
    # Create a ChatProcessor and run
    processor = ChatProcessor()
    
    # Example with default settings
    processor.chat_with_data()
    
    # Example with custom output directory
    # processor.chat_with_data(output_dir="custom_output")
    
    # Example with chat model override
    # processor.chat_with_data(
    #     chat_model_override={
    #         "MODEL_NAME": "DEEPSEEK-R1-QWEN-14B",
    #         "NUM_TOKENS_TO_OUTPUT": 64000,
    #         "TEMPERATURE": 0.5
    #     }
    # )