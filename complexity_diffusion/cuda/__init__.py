"""
CUDA/Triton kernels for Complexity Diffusion (optional).
"""

HAS_TRITON = False

try:
    import triton
    HAS_TRITON = True
except ImportError:
    pass

__all__ = ["HAS_TRITON"]
