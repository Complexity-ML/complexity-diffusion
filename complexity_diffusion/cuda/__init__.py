"""
CUDA/Triton kernels for Complexity Diffusion.

Provides accelerated implementations of:
- INL Dynamics (fused controller + dynamics equations)
- RMSNorm (fused normalization)
- Flash Attention (memory-efficient attention)
"""

HAS_TRITON = False

try:
    import triton
    HAS_TRITON = True
except ImportError:
    pass

# Only import accelerated functions if Triton is available
if HAS_TRITON:
    from .triton_dynamics import (
        inl_dynamics,
        triton_inl_dynamics,
        pytorch_inl_dynamics,
    )
    from .triton_norm import (
        rms_norm,
        triton_rms_norm,
        pytorch_rms_norm,
    )
    from .triton_attention import (
        attention,
        triton_flash_attention,
        pytorch_attention,
    )
else:
    # Fallback: import only PyTorch versions
    from .triton_dynamics import pytorch_inl_dynamics as inl_dynamics
    from .triton_dynamics import pytorch_inl_dynamics
    from .triton_norm import pytorch_rms_norm as rms_norm
    from .triton_norm import pytorch_rms_norm
    from .triton_attention import pytorch_attention as attention
    from .triton_attention import pytorch_attention
    
    # Dummy triton functions that just call pytorch
    triton_inl_dynamics = pytorch_inl_dynamics
    triton_rms_norm = pytorch_rms_norm
    triton_flash_attention = pytorch_attention

__all__ = [
    "HAS_TRITON",
    # Dynamics
    "inl_dynamics",
    "triton_inl_dynamics",
    "pytorch_inl_dynamics",
    # Normalization
    "rms_norm",
    "triton_rms_norm",
    "pytorch_rms_norm",
    # Attention
    "attention",
    "triton_flash_attention",
    "pytorch_attention",
]
