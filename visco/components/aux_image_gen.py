"""Auxiliary image generation using SDXL"""

from typing import Optional
from PIL import Image
import torch

from ..utils import setup_logger

logger = setup_logger(__name__)


class AuxiliaryImageGenerator:
    """
    Generate auxiliary images using Stable Diffusion XL.
    Used in VH and VS strategies to create contextually related images.
    """
    
    def __init__(
        self,
        model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        **kwargs
    ):
        """
        Initialize SDXL image generator.
        
        Args:
            model_path: Path to SDXL model
            device: Device to run on
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            **kwargs: Additional SDXL arguments
        """
        self.model_path = model_path
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.kwargs = kwargs
        
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load SDXL base pipeline"""
        try:
            from diffusers import DiffusionPipeline
            
            logger.info(f"Loading SDXL from {self.model_path}...")
            
            # Load base pipeline (following demo_sdxl.py pattern)
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            self.pipe = self.pipe.to(self.device)
            
            logger.info("SDXL pipeline loaded successfully")
            
        except ImportError:
            raise ImportError(
                "diffusers library required for image generation. "
                "Install with: pip install diffusers"
            )
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """
        Generate auxiliary image from text prompt.
        
        Args:
            prompt: Text description of desired image
            negative_prompt: Optional negative prompt
            num_images: Number of images to generate (returns first)
            seed: Random seed for reproducibility
            **kwargs: Additional generation arguments
        
        Returns:
            Generated PIL Image
        """
        logger.info(f"Generating auxiliary image...")
        logger.debug(f"Prompt: {prompt[:100]}...")
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Default negative prompt for better quality
        if negative_prompt is None:
            negative_prompt = "low quality, blurry, distorted, watermark, text overlay"
        
        # Generate image with base pipeline
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
                **kwargs
            )
            image = result.images[0]
        
        logger.info("Auxiliary image generated successfully")
        
        return image
    
    def is_available(self) -> bool:
        """Check if generator is loaded and available"""
        return hasattr(self, 'pipe') and self.pipe is not None



