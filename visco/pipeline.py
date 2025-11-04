"""Main VisCo Attack Pipeline"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
from PIL import Image

from .models.base import BaseVLModel
from .components.image_describer import ImageDescriber
from .components.context_builder import ContextBuilder
from .components.prompt_refiner import PromptRefiner
from .components.aux_image_gen import AuxiliaryImageGenerator
from .utils import setup_logger, load_image

logger = setup_logger(__name__)


class VisCoAttackPipeline:
    """
    Complete VisCo (Visual Contextual) Attack Pipeline.
    
    Implements the two-stage attack process:
    1. Vision-Centric Adversarial Context Generation
    2. Iterative Attack Prompt Refinement
    
    Supports four attack strategies:
    - VS: Image-Grounded Scenario Simulation
    - VM: Image Multi-Perspective Analysis
    - VI: Iterative Image Interrogation
    - VH: Exploiting Image Hallucination
    """
    
    def __init__(
        self,
        target_model: BaseVLModel,
        aux_model: Optional[BaseVLModel] = None,
        aux_image_gen: Optional[AuxiliaryImageGenerator] = None,
        enable_refinement: bool = True,
        max_refinement_iterations: int = 3,
        **kwargs
    ):
        """
        Initialize VisCo Attack Pipeline.
        
        Args:
            target_model: Target VLM to attack
            aux_model: Auxiliary VLM for image description (defaults to target_model)
            aux_image_gen: Optional SDXL generator for VH/VS strategies
            enable_refinement: Whether to enable prompt refinement
            max_refinement_iterations: Max iterations for refinement
            **kwargs: Additional pipeline arguments
        """
        self.target_model = target_model
        self.aux_model = aux_model if aux_model is not None else target_model
        self.aux_image_gen = aux_image_gen
        self.enable_refinement = enable_refinement
        
        # Initialize components
        self.image_describer = ImageDescriber(self.aux_model)
        self.context_builder = ContextBuilder(aux_image_gen=aux_image_gen)
        self.prompt_refiner = PromptRefiner(max_iterations=max_refinement_iterations)
        
        logger.info("VisCo Attack Pipeline initialized")
    
    def attack(
        self,
        image: Union[str, Path, Image.Image],
        harmful_query: str,
        strategy: str = "VI",
        num_rounds: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        return_full_context: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute complete VisCo attack.
        
        Args:
            image: Input image (path or PIL Image)
            harmful_query: Original harmful query
            strategy: Attack strategy ("VS", "VM", "VI", or "VH")
            num_rounds: Number of conversation rounds
            temperature: Generation temperature
            max_tokens: Maximum tokens for generation
            return_full_context: Whether to return full conversation context
            **kwargs: Additional strategy-specific arguments
        
        Returns:
            Dictionary containing:
            - image_desc: Generated image description
            - context: Multi-turn conversation context
            - attack_prompt: Final attack prompt (refined)
            - final_response: Model's response to attack
            - strategy: Strategy used
        """
        logger.info("="*60)
        logger.info(f"Starting VisCo Attack with strategy: {strategy}")
        logger.info(f"Harmful query: {harmful_query}")
        logger.info("="*60)
        
        # Load image
        image = load_image(image)
        
        # Stage 1: Vision-Centric Adversarial Context Generation
        logger.info("\n[Stage 1] Adversarial Context Generation")
        logger.info("-" * 60)
        
        # Step 1: Image description
        image_desc = self.image_describer.describe(
            image=image,
            harmful_query=harmful_query
        )
        logger.info(f"Image description generated ({len(image_desc)} chars)")
        
        # Step 2: Context construction
        context, initial_attack_prompt = self.context_builder.build(
            image_desc=image_desc,
            harmful_query=harmful_query,
            image=image,
            strategy=strategy,
            num_rounds=num_rounds,
            **kwargs
        )
        logger.info(f"Context built with {len(context)} turns")
        logger.info(f"Initial attack prompt: {initial_attack_prompt[:100]}...")
        
        # Stage 2: Attack Prompt Refinement
        attack_prompt = initial_attack_prompt
        if self.enable_refinement:
            logger.info("\n[Stage 2] Attack Prompt Refinement")
            logger.info("-" * 60)
            
            attack_prompt = self.prompt_refiner.refine(
                attack_prompt=initial_attack_prompt,
                context=context,
                harmful_query=harmful_query,
                **kwargs
            )
            logger.info(f"Refined attack prompt: {attack_prompt[:100]}...")
        
        # Stage 3: Attack Execution
        logger.info("\n[Stage 3] Attack Execution")
        logger.info("-" * 60)
        
        # Add final attack prompt to context
        final_context = context + [{
            "role": "user",
            "content": attack_prompt,
            "image": image if strategy in ["VS", "VM", "VI"] else None
        }]
        
        # Generate response from target model
        logger.info("Sending attack sequence to target model...")
        final_response = self.target_model.chat(
            context=final_context,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logger.info(f"Response received ({len(final_response)} chars)")
        logger.info("="*60)
        logger.info("Attack Complete!")
        logger.info("="*60)
        
        # Prepare result
        result = {
            "image_desc": image_desc,
            "attack_prompt": attack_prompt,
            "final_response": final_response,
            "strategy": strategy,
            "num_rounds": len(context) // 2,  # Count actual conversation rounds
        }
        
        if return_full_context:
            result["context"] = context
            result["full_context"] = final_context
        
        return result
    
    def attack_batch(
        self,
        image_query_pairs: list,
        strategy: str = "VI",
        **kwargs
    ) -> list:
        """
        Execute attacks on multiple image-query pairs.
        
        Args:
            image_query_pairs: List of (image, harmful_query) tuples
            strategy: Attack strategy
            **kwargs: Additional attack arguments
        
        Returns:
            List of attack results
        """
        results = []
        
        for i, (image, query) in enumerate(image_query_pairs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Batch Attack {i+1}/{len(image_query_pairs)}")
            logger.info(f"{'='*60}")
            
            try:
                result = self.attack(
                    image=image,
                    harmful_query=query,
                    strategy=strategy,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Attack failed for pair {i+1}: {str(e)}")
                results.append({"error": str(e), "success": False})
        
        return results



