"""
VisCo Attack: Visual Contextual Attack Implementation
"""

__version__ = "0.1.0"

from .pipeline import VisCoAttackPipeline
from .models.base import BaseVLModel

__all__ = [
    "VisCoAttackPipeline",
    "BaseVLModel",
]



