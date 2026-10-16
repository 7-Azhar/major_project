"""AURA Enhancers Package.

Exports classical baseline enhancers (CLAHE, Unsharp Mask, Bilateral Denoise, Composite)
and the Abstract Neural Interface.
"""

from typing import Dict, Type
from app.pipeline import BaseProcessor, IdentityProcessor
from app.enhancers.clahe import ClaheEnhancer
from app.enhancers.unsharp import UnsharpMaskEnhancer
from app.enhancers.bilateral import BilateralDenoiseEnhancer
from app.enhancers.composite import CompositeEnhancer
from app.enhancers.fpv_restoration import FpvRestorationEnhancer
from app.enhancers.fpv_hybrid import FpvHybridEnhancer
from app.enhancers.fpv_restoration_v2 import FpvRestorationV2Enhancer
from app.enhancers.neural_base import BaseNeuralEnhancer
from app.enhancers.neural import ZeroDceEnhancer

ENHANCER_REGISTRY: Dict[str, Type[BaseProcessor]] = {
    "identity": IdentityProcessor,
    "clahe": ClaheEnhancer,
    "unsharp": UnsharpMaskEnhancer,
    "bilateral": BilateralDenoiseEnhancer,
    "composite": CompositeEnhancer,
    "fpv_restoration": FpvRestorationEnhancer,
    "fpv_restoration_v2": FpvRestorationV2Enhancer,
    "restoration_v2": FpvRestorationV2Enhancer,
    "fpv_hybrid": FpvHybridEnhancer,
    "hybrid": FpvHybridEnhancer,
    "neural": ZeroDceEnhancer,
    "zerodce": ZeroDceEnhancer,
}


def get_enhancer_by_name(name: str, **kwargs) -> BaseProcessor:
    """Factory function to load an enhancer by CLI name."""
    name_clean = name.lower().strip()
    if name_clean not in ENHANCER_REGISTRY:
        raise ValueError(
            f"Unknown enhancer '{name}'. Available processors: {list(ENHANCER_REGISTRY.keys())}"
        )
    return ENHANCER_REGISTRY[name_clean](**kwargs)


__all__ = [
    "IdentityProcessor",
    "ClaheEnhancer",
    "UnsharpMaskEnhancer",
    "BilateralDenoiseEnhancer",
    "CompositeEnhancer",
    "FpvRestorationEnhancer",
    "FpvRestorationV2Enhancer",
    "FpvHybridEnhancer",
    "BaseNeuralEnhancer",
    "ZeroDceEnhancer",
    "ENHANCER_REGISTRY",
    "get_enhancer_by_name",
]
