"""
Triton-accelerated RMSNorm kernel.

Fuses normalization and scaling into a single kernel.
"""

import torch
from typing import Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _rms_norm_fwd_kernel(
        x_ptr, weight_ptr, out_ptr, rstd_ptr,
        N: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Forward RMSNorm kernel."""
        row = tl.program_id(0)

        # Compute offset for this row
        row_start = row * N

        # Compute mean of squares
        _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            _sum_sq += x * x

        mean_sq = tl.sum(_sum_sq) / N
        rstd = 1.0 / tl.sqrt(mean_sq + eps)

        # Store rstd for backward
        tl.store(rstd_ptr + row, rstd)

        # Normalize and scale
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            out = x * rstd * w
            tl.store(out_ptr + row_start + cols, out, mask=mask)


    @triton.jit
    def _rms_norm_bwd_kernel(
        dy_ptr, x_ptr, weight_ptr, rstd_ptr,
        dx_ptr, dw_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Backward RMSNorm kernel."""
        row = tl.program_id(0)
        row_start = row * N

        rstd = tl.load(rstd_ptr + row)

        # Compute gradient contributions
        _sum_dy_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            _sum_dy_x += dy * w * x

        sum_dy_x = tl.sum(_sum_dy_x)

        # Compute dx
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

            # dx = rstd * w * (dy - x * rstd^2 * sum(dy * w * x) / N)
            dx = rstd * w * (dy - x * rstd * rstd * sum_dy_x / N)
            tl.store(dx_ptr + row_start + cols, dx, mask=mask)

            # Accumulate dw (needs atomic or separate reduction)
            dw_contrib = dy * x * rstd
            tl.atomic_add(dw_ptr + cols, dw_contrib, mask=mask)


class TritonRMSNormFunction(torch.autograd.Function):
    """Autograd function for Triton RMSNorm."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        # Flatten to 2D
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()

        n_rows, N = x_2d.shape
        out = torch.empty_like(x_2d)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)

        BLOCK_SIZE = 128
        grid = (n_rows,)

        _rms_norm_fwd_kernel[grid](
            x_2d, weight, out, rstd,
            N=N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.orig_shape = orig_shape
        ctx.N = N

        return out.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_2d, weight, rstd = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        N = ctx.N

        dy_2d = dy.reshape(-1, N).contiguous()
        n_rows = dy_2d.shape[0]

        dx = torch.empty_like(x_2d)
        dw = torch.zeros_like(weight)

        BLOCK_SIZE = 128
        grid = (n_rows,)

        _rms_norm_bwd_kernel[grid](
            dy_2d, x_2d, weight, rstd,
            dx, dw,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return dx.reshape(orig_shape), dw, None


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton-accelerated RMSNorm."""
    return TritonRMSNormFunction.apply(x, weight, eps)


def pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pure PyTorch RMSNorm (fallback)."""
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(mean_sq + eps)
    return x_norm * weight


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    use_triton: Optional[bool] = None,
) -> torch.Tensor:
    """RMSNorm with automatic backend selection."""
    if use_triton is None:
        use_triton = HAS_TRITON and x.is_cuda

    if use_triton and HAS_TRITON:
        return triton_rms_norm(x, weight, eps)
    else:
        return pytorch_rms_norm(x, weight, eps)
