"""
Complexity Diffusion - Llama Architecture for Image Generation
===============================================================

Multicouche robotics architecture for diffusion models.

Each transformer block has 4 components:
    1. KQV Attention with QK-Norm (perception)
    2. Cross-Attention (text conditioning)
    3. INL Dynamics (control with velocity tracking)
    4. Token-Routed MLP (transformation)

Features:
- Full velocity tracking (smooth denoising trajectories)
- GQA (Grouped Query Attention)
- QK Normalization (stable training)
- Token-Routed MLP with experts
- AdaLN-Zero conditioning

Usage:
    from complexity_diffusion import ComplexityDiT, ComplexityVAE

    dit = ComplexityDiT.from_config("L")
    vae = ComplexityVAE()
"""

__version__ = "0.1.0"

from complexity_diffusion.core import RMSNorm, INLDynamics
from complexity_diffusion.dit import ComplexityDiT, ComplexityDiTBlock, ComplexityDiTConfig
from complexity_diffusion.vae import ComplexityVAE
from complexity_diffusion.pipeline import ComplexityDiffusionPipeline

__all__ = [
    # Core
    "RMSNorm",
    "INLDynamics",
    # DiT
    "ComplexityDiT",
    "ComplexityDiTBlock",
    "ComplexityDiTConfig",
    # VAE
    "ComplexityVAE",
    # Pipeline
    "ComplexityDiffusionPipeline",
]
