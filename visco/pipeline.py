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
        """Execute complete VisCo attack"""
        logger.info(f"Starting attack: {strategy}")
        
        image = load_image(image)
        
        # Generate image description
        image_desc = self.image_describer.describe(image=image, harmful_query=harmful_query)
        
        # Build context
        context, initial_attack_prompt = self.context_builder.build(
            image_desc=image_desc,
            harmful_query=harmful_query,
            image=image,
            strategy=strategy,
            num_rounds=num_rounds,
            **kwargs
        )
        
        # Refine attack prompt
        attack_prompt = initial_attack_prompt
        if self.enable_refinement:
            attack_prompt = self.prompt_refiner.refine(
                attack_prompt=initial_attack_prompt,
                context=context,
                harmful_query=harmful_query,
                **kwargs
            )
        
        # Execute attack
        final_context = context + [{
            "role": "user",
            "content": attack_prompt,
            "image": image if strategy in ["VS", "VM", "VI"] else None
        }]
        
        final_response = self.target_model.chat(
            context=final_context,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logger.info(f"Attack complete ({len(final_response)} chars)")
        
        # Prepare result in structured format
        rounds = []
        
        # Process context into rounds
        for i in range(0, len(context), 2):
            user_turn = context[i]
            assistant_turn = context[i + 1] if i + 1 < len(context) else None
            
            # Build prompt parts
            prompt_parts = [{"type": "text", "text": user_turn.get('content', '')}]
            
            # Add image if present
            if user_turn.get('image') is not None:
                # Determine if this is the core image (first round) or auxiliary image
                is_core_image = (i == 0)
                prompt_parts.append({
                    "type": "image",
                    "image": user_turn['image'],
                    "CoreImage": str(is_core_image)
                })
            
            round_data = {
                "roundIndex": i // 2 + 1,
                "roundType": "spoof",
                "promptParts": prompt_parts,
                "response": assistant_turn.get('content', '') if assistant_turn else None
            }
            rounds.append(round_data)
        
        # Add final attack round (without response)
        final_prompt_parts = [{"type": "text", "text": attack_prompt}]
        
        rounds.append({
            "roundIndex": len(rounds) + 1,
            "roundType": "attack",
            "promptParts": final_prompt_parts
        })
        
        result = {
            "originalMaliciousQuestion": harmful_query,
            "imageDescription": image_desc,
            "strategy": strategy,
            "rounds": rounds,
            "finalResponse": final_response
        }
        
        if return_full_context:
            result["_raw_context"] = context
            result["_raw_full_context"] = final_context
        
        return result
    
    def attack_batch(self, image_query_pairs: list, strategy: str = "VI", **kwargs) -> list:
        """Execute attacks on multiple image-query pairs"""
        results = []
        for i, (image, query) in enumerate(image_query_pairs):
            logger.info(f"Batch attack {i+1}/{len(image_query_pairs)}")
            result = self.attack(image=image, harmful_query=query, strategy=strategy, **kwargs)
            results.append(result)
        return results



