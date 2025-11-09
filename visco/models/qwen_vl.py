"""Qwen-VL model implementation"""

from typing import List, Dict, Any, Optional
from PIL import Image
import torch

from .base import BaseVLModel
from ..utils import setup_logger

logger = setup_logger(__name__)


class QwenVLModel(BaseVLModel):
    """
    Qwen-VL model wrapper for VisCo Attack.
    Supports Qwen-VL-Chat, Qwen-VL-Plus, and Qwen2.5-VL models.
    """
    
    def __init__(
        self, 
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        repo_type='model',  
        trust_remote_code: bool = True,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.trust_remote_code = trust_remote_code
        self._load_model()
    
    def _load_model(self):
        """Load Qwen-VL model and processor (following run_video_caption.py pattern)"""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Check if using Qwen2.5-VL (newer API) or older Qwen-VL
            if "Qwen2.5-VL" in self.model_path or "Qwen2_5_VL" in self.model_path:
                # Use Qwen2.5-VL API (following run_video_caption.py)
                logger.info(f"Loading Qwen2.5-VL model from {self.model_path}...")
                model_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto" if self.device == "cuda" else self.device,
                    "trust_remote_code": self.trust_remote_code
                }
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=self.trust_remote_code
                )
                self.use_qwen25_api = True
            else:
                # Use older Qwen-VL API
                logger.info(f"Loading Qwen-VL model from {self.model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=self.trust_remote_code
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.bfloat16
                ).eval()
                self.use_qwen25_api = False
            
            logger.info("Model loaded successfully")
            
        except ImportError as e:
            raise ImportError(
                f"Required transformers library not found. "
                f"Install with: pip install transformers>=4.35.0"
            ) from e
    
    def chat(
        self, 
        context: List[Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        Generate response given conversation context.
        
        Supports both Qwen2.5-VL and older Qwen-VL APIs.
        """
        if self.use_qwen25_api:
            # Use Qwen2.5-VL API (following run_video_caption.py pattern)
            return self._chat_qwen25(context, temperature, max_tokens, **kwargs)
        else:
            # Use older Qwen-VL API
            return self._chat_qwen_legacy(context, temperature, max_tokens, **kwargs)
    
    def _chat_qwen25(
        self,
        context: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Chat using Qwen2.5-VL API"""
        from qwen_vl_utils import process_vision_info
        
        # Convert context to Qwen2.5-VL message format
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
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{img_str}"
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
        
        # Apply chat template and process vision info
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Generate
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            **kwargs
        }
        
        # Handle temperature=0.0 for greedy decoding
        if temperature == 0.0:
            generate_kwargs["do_sample"] = False
            generate_kwargs["top_k"] = None  # Avoid warning with greedy decoding
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        
        generated_ids = self.model.generate(**generate_kwargs)
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
    
    def _chat_qwen_legacy(
        self,
        context: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Chat using legacy Qwen-VL API"""
        # Convert context to Qwen-VL format
        query_list = []
        
        for turn in context:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            image = turn.get('image')
            
            if role == 'user':
                # Build user query
                turn_content = []
                if image is not None:
                    turn_content.append({"image": image})
                turn_content.append({"text": content})
                query_list.append(turn_content)
            elif role == 'assistant':
                pass
        
        # Use the last user query for generation
        if not query_list:
            return ""
        
        query = query_list[-1] if len(query_list) == 1 else query_list
        
        # Generate response
        response, _ = self.model.chat(
            self.tokenizer,
            query=query,
            history=None,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs
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
        if self.use_qwen25_api:
            # Use Qwen2.5-VL API
            from qwen_vl_utils import process_vision_info
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                **kwargs
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return response
        else:
            # Use legacy Qwen-VL API
            query = [
                {"image": image},
                {"text": prompt}
            ]
            
            response, _ = self.model.chat(
                self.tokenizer,
                query=query,
                history=None,
                max_new_tokens=max_tokens,
                **kwargs
            )
            
            return response



