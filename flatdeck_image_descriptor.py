import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio
import copy

from llama_vision_connector import LlamaVisionConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageDescriptor:
    """
    A class for processing extracted PDF content using LLM models.
    
    This processor takes the output of PDFIngestor and enriches it with
    LLM-generated descriptions of images and content.
    """
    
    def __init__(self, config_path: str = "config/config.json", 
                 config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM processor.
        
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
    
    def _build_vision_model_config(self, vision_model_override: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Build a vision model configuration by combining models.json and config.json settings.
        
        Args:
            vision_model_override: Optional dictionary to override the DEFAULT_VISION_MODEL settings
            
        Returns:
            Dictionary with model configuration compatible with LlamaVisionConnector
        """
        # Load the base models configuration
        base_models_config = self._load_models_config()
        
        # Get the vision model settings, with override if provided
        vision_model_config = vision_model_override if vision_model_override else self.config.get("DEFAULT_VISION_MODEL", {})
        model_name = vision_model_config.get("MODEL_NAME", "GEMMA3_12B")
        
        # Make a deep copy of the original models configuration
        models_config = copy.deepcopy(base_models_config.get("MODELS", {}))
        
        # Ensure the target model exists in the configuration
        if model_name not in models_config:
            logger.warning(f"Model {model_name} not found in models.json. Creating new entry.")
            models_config[model_name] = {}
        
        # Update the model configuration with settings from config.json
        for key, value in vision_model_config.items():
            if key != "MODEL_NAME":  # Skip the MODEL_NAME as it's not a model parameter
                models_config[model_name][key] = value
        
        return models_config
    
    async def describe_images(self, output_dir: Optional[str] = None, 
                          describe_all: bool = False,
                          vision_model_override: Optional[Dict[str, Any]] = None) -> None:
        """
        Generate descriptions for images using a vision model and update the corresponding JSON files.
        
        Args:
            output_dir: Optional path to the output directory (overrides the one in config)
            describe_all: If True, describe all images (page and embedded images), if False, describe only page images
            vision_model_override: Optional dictionary to override the DEFAULT_VISION_MODEL settings,
                                  should include at least a "MODEL_NAME" key
        """
        # Use provided output directory or default from config
        output_path = Path(output_dir) if output_dir else self.output_dir
        
        if not output_path.exists():
            logger.error(f"Output directory {output_path} does not exist")
            raise FileNotFoundError(f"Output directory {output_path} does not exist")
        
        # Get vision prompt path from config
        vision_prompt_filename = self.config.get("VISION_PROMPT_FILENAME", "vision-prompt.txt")
        vision_prompt_path = Path(vision_prompt_filename)
        
        # Read the vision prompt if available
        vision_prompt = None
        if vision_prompt_path.exists():
            try:
                with open(vision_prompt_path, 'r') as f:
                    vision_prompt = f.read()
                    logger.info(f"Loaded vision prompt from {vision_prompt_path}")
            except Exception as e:
                logger.warning(f"Could not read vision prompt from {vision_prompt_path}: {str(e)}")
        
        # Get vision model name from config, with override if provided
        vision_model_config = vision_model_override if vision_model_override else self.config.get("DEFAULT_VISION_MODEL", {})
        model_name = vision_model_config.get("MODEL_NAME", "GEMMA3_12B")
        
        # Get model configuration by merging models.json with config.json settings
        models_config = self._build_vision_model_config(vision_model_override)
        
        # Use the combined configuration
        models_config_override = {
            "MODELS": models_config
        }
        
        # Initialize the LlamaVisionConnector with our model configuration
        try:
            # Create a custom config.json with our model settings
            temp_config_path = output_path / "temp_vision_config.json"
            with open(temp_config_path, 'w') as f:
                json.dump(models_config_override, f, indent=2)
            
            # Initialize vision connector with our temporary config
            model_name = self.config.get("DEFAULT_VISION_MODEL", {}).get("MODEL_NAME", "GEMMA3_12B")
            vision_connector = LlamaVisionConnector(
                config_path=str(temp_config_path),
                model_key=model_name
            )
            
            # Clean up temporary config file
            if temp_config_path.exists():
                os.remove(temp_config_path)
                
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
                    
                    # Process page image if available
                    if "page_image" in page_data and page_data["page_image"]:
                        page_image_path = output_path / page_data["page_image"]
                        
                        if page_image_path.exists():
                            logger.info(f"Describing page image: {page_image_path}")
                            
                            # Get image description from vision model
                            description = await vision_connector.get_response(
                                str(page_image_path),
                                prompt=vision_prompt
                            )
                            
                            if description:
                                # Update page description in the JSON data
                                page_data["content"]["page_description"] = description
                                logger.info(f"Added page description for {file_id} page {page_number}")
                            else:
                                logger.warning(f"Failed to get description for page image: {page_image_path}")
                    
                    # Process embedded images if describe_all is True
                    if describe_all and "content" in page_data and "image_data" in page_data["content"]:
                        for i, image_info in enumerate(page_data["content"]["image_data"]):
                            if "image_filename" in image_info:
                                image_path = output_path / image_info["image_filename"]
                                
                                if image_path.exists():
                                    logger.info(f"Describing embedded image {i+1}/{len(page_data['content']['image_data'])}: {image_path}")
                                    
                                    # Get image description from vision model
                                    description = await vision_connector.get_response(
                                        str(image_path),
                                        prompt=vision_prompt
                                    )
                                    
                                    if description:
                                        # Update image description in the JSON data
                                        image_info["image_description"] = description
                                        logger.info(f"Added description for embedded image {image_info['image_filename']}")
                                    else:
                                        logger.warning(f"Failed to get description for embedded image: {image_path}")
                    
                    # Save updated JSON data back to file
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(page_data, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {str(e)}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Error initializing vision connector: {str(e)}", exc_info=True)
            raise
                

async def main():
    """Main function to demonstrate the image description process."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate descriptions for images in the output directory.")
    parser.add_argument("--output", "-o", help="Output directory containing images to describe", default="output")
    parser.add_argument("--all", "-a", action="store_true", help="Describe all images, including embedded ones", default=False)
    
    args = parser.parse_args()
    
    # Create an image descriptor and run
    descriptor = ImageDescriptor()
    
    # Call describe_images with appropriate parameters
    await descriptor.describe_images(
        output_dir=args.output,
        describe_all=args.all
    )
    
    print(f"Image description completed successfully for directory: {args.output}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())