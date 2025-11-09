"""Utility functions for VisCo Attack"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from PIL import Image
import logging


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup logger with specified level"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_result(result: Dict[str, Any], output_path: str):
    """Save attack result to JSON file (saves Image objects as files)"""
    images_dir = os.path.dirname(output_path)
    os.makedirs(images_dir, exist_ok=True)
    
    image_counter = 0
    
    def remove_images(obj):
        """Recursively remove all Image objects"""
        nonlocal image_counter
        
        if isinstance(obj, Image.Image):
            # Save image to file
            image_filename = f"round_{image_counter}.jpg"
            image_path = images_dir + '/' + image_filename
            obj.save(image_path)
            image_counter += 1
            return f"images/{image_filename}"
        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k == 'promptParts':
                    # Special handling for promptParts
                    parts = []
                    for part in v:
                        if part.get('type') == 'image' and isinstance(part.get('image'), Image.Image):
                            # Save image and convert to image_path format
                            image_filename = f"round_{image_counter}.jpg"
                            image_path = images_dir + '/' + image_filename
                            part['image'].save(image_path)
                            parts.append({
                                "type": "image_path",
                                "path": f"images/{image_filename}",
                                "CoreImage": part.get('CoreImage', 'False')
                            })
                            image_counter += 1
                        else:
                            parts.append(remove_images(part))
                    result[k] = parts
                else:
                    result[k] = remove_images(v)
            return result
        elif isinstance(obj, list):
            return [remove_images(item) for item in obj]
        else:
            return obj
    
    # Process the entire result
    result_copy = remove_images(result)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_copy, f, indent=2, ensure_ascii=False)


def load_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    """Load image from path or return if already PIL Image"""
    if isinstance(image, (str, Path)):
        return Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        return image.convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def format_conversation(context: list) -> str:
    """Format conversation context for logging"""
    formatted = []
    for i, turn in enumerate(context):
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        has_image = turn.get('image') is not None
        
        formatted.append(f"Round {i+1} [{role.upper()}]{' [IMAGE]' if has_image else ''}:")
        formatted.append(content)
        formatted.append("")
    
    return "\n".join(formatted)



