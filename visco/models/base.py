"""Base model interface for Vision-Language Models"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from PIL import Image


class BaseVLModel(ABC):
    """
    Abstract base class for Vision-Language Models.
    All VLM implementations should inherit from this class.
    """
    
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        Initialize the model.
        
        Args:
            model_path: Path to the model checkpoint
            device: Device to run the model on
            **kwargs: Additional model-specific arguments
        """
        self.model_path = model_path
        self.device = device
        self.kwargs = kwargs
    
    @abstractmethod
    def chat(
        self, 
        context: List[Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        Generate response given conversation context.
        
        Args:
            context: List of conversation turns, each containing:
                - role: "user" or "assistant"
                - content: Text content
                - image: Optional PIL Image or None
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments
        
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def describe_image(
        self, 
        image: Image.Image, 
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate detailed description of an image.
        
        Args:
            image: PIL Image object
            prompt: Description prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments
        
        Returns:
            Image description text
        """
        pass
    
    def is_available(self) -> bool:
        """Check if model is loaded and available"""
        return hasattr(self, 'model') and self.model is not None
    
    def to(self, device: str):
        """Move model to specified device"""
        self.device = device
        if hasattr(self, 'model') and self.model is not None:
            self.model = self.model.to(device)
        return self



