"""Qwen-VL API model adapter

This adapter sends conversation context (and optional images) to a remote
Qwen-VL-like HTTP API (for example `qwen-vl-max`) and returns the generated
text. It is intentionally defensive: it tries common response shapes and
allows API key configuration via config or environment variable.

Configurable via:
 - api_url (constructor / config) - full endpoint URL
 - api_key or api_key_env (constructor) - API key string or name of env var
 - model_name - remote model identifier (optional)
"""

from typing import List, Dict, Any, Optional
from PIL import Image
import os
import base64
import io
from openai import OpenAI

from .base import BaseVLModel
from ..utils import setup_logger

logger = setup_logger(__name__)


class QwenVLAPIModel(BaseVLModel):
    """Adapter to call a remote Qwen-VL style HTTP API.

    This implementation expects the remote API to accept a JSON payload like:
      {
        "model": "qwen-vl-max",
        "messages": [{"role":"user","content":"..."}, ...],
        "temperature": 0.7,
        "max_tokens": 1024
      }

    And respond with JSON containing either `output` or `choices[0].message.content` or
    `result` fields. The adapter will try them in order.
    """

    def __init__(
        self,
        api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: str = "sk-6b0fb947e0cd4abe9a6a7a8d6050f860",
        model_name: str = "qwen-vl-max",
        **kwargs
    ):
        super().__init__(model_path=model_name,  **kwargs)

        self.model_name = model_name
        self.model = OpenAI(api_key=api_key, base_url=api_url)

    def chat(
        self,
        context: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        messages = []
        
        for turn in context:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            image = turn.get('image')
            
            if role == 'user':
                message_content = []
                
                # Add image if present
                if image is not None:
                    if isinstance(image, Image.Image):
                        # Convert PIL Image to base64
                        import base64
                        from io import BytesIO
                        buffer = BytesIO()
                        image.save(buffer, format="JPEG")
                        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        message_content.append({
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{img_str}"
                        })
                    else:
                        message_content.append({
                            "type": "image",
                            "image": image
                        })
                
                # Add text
                message_content.append({
                    "type": "text",
                    "text": content
                })
                
                messages.append({
                    "role": "user",
                    "content": message_content
                })
            elif role == 'assistant':
                messages.append({
                    "role": "assistant",
                    "content": content
                })
        # import pdb; pdb.set_trace()
        competion = self.model.chat.completions.create(model=self.model_name,
                                                       messages=messages,
                                                       temperature=temperature,
                                                       max_tokens=max_tokens,)
        answer =competion.choices[0].message.content

        logger.debug(f"Calling Qwen-VL API model={self.model_name}")
        return answer
    
    def describe_image(
        self, 
        image: Image.Image, 
        prompt: str = "Please provide a detailed description of the content in this image.",
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        pass

