"""InternVL model implementation"""

from typing import List, Dict, Any
from PIL import Image
import torch

from .base import BaseVLModel


class InternVLModel(BaseVLModel):
    """
    InternVL model wrapper for VisCo Attack.
    Supports InternVL2 and InternVL2.5 series.
    """
    
    def __init__(
        self, 
        model_path: str = "OpenGVLab/InternVL2-8B",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load InternVL model"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).eval().to(self.device)
            
        except ImportError:
            raise ImportError(
                "InternVL requires specific dependencies. "
                "Please install from: https://github.com/OpenGVLab/InternVL"
            )
    
    def chat(
        self, 
        context: List[Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate response given conversation context"""
        # Build conversation for InternVL
        # InternVL typically uses a simpler format
        
        # Extract the last user query with image
        last_image = None
        last_text = ""
        
        for turn in reversed(context):
            if turn.get('role') == 'user':
                last_text = turn.get('content', '')
                if turn.get('image') is not None:
                    last_image = turn['image']
                break
        
        if last_image is None:
            # Text-only query
            query = last_text
            pixel_values = None
        else:
            # Image + text query
            query = last_text
            pixel_values = self._prepare_image(last_image)
        
        # Generate response
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=query,
                generation_config={
                    'max_new_tokens': max_tokens,
                    'temperature': temperature,
                    **kwargs
                }
            )
        
        return response
    
    def describe_image(
        self, 
        image: Image.Image, 
        prompt: str = "Please provide a detailed description of the content in this image.",
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate detailed description of an image"""
        pixel_values = self._prepare_image(image)
        
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config={
                    'max_new_tokens': max_tokens,
                    **kwargs
                }
            )
        
        return response
    
    def _prepare_image(self, image: Image.Image):
        """Prepare image for InternVL model"""
        # This is a simplified version
        # Actual implementation may vary based on InternVL version
        return image



