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
    """Save attack result to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


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



