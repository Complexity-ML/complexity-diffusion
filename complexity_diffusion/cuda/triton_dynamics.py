"""
Triton-accelerated INL Dynamics kernel.

Fuses the entire dynamics computation into a single kernel:
    error = h - mu
    v_next = alpha * v - beta * error
    h_next = h + dt * gate * v_next
"""

import torch
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


# Fallback PyTorch implementation (always available)
def pytorch_inl_dynamics(
    h: torch.Tensor,
    v: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    dt: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch implementation (fallback)."""
    error = h - mu
    v_next = alpha * v - beta * error
    h_next = h + dt * gate * v_next
    return h_next, v_next


if HAS_TRITON:
    @triton.jit
    def _inl_dynamics_kernel(
        # Input pointers
        h_ptr, v_ptr, mu_ptr,
        alpha_ptr, beta_ptr, gate_ptr,
        # Output pointers
        h_out_ptr, v_out_ptr,
        # Total number of elements
        n_total,
        # Hidden size (for mu broadcasting)
        hidden_size,
        # Scalar
        dt,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fully parallelized dynamics kernel.
        Each program processes BLOCK_SIZE elements across the entire flattened tensor.
        """
        pid = tl.program_id(0)

        # Global offsets for this block
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_total

        # Compute mu index (wraps around hidden_size)
        mu_idx = offsets % hidden_size

        # Load all inputs
        h = tl.load(h_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
        mu = tl.load(mu_ptr + mu_idx, mask=mask, other=0.0)
        alpha = tl.load(alpha_ptr + offsets, mask=mask, other=0.0)
        beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)

        # Fused dynamics computation
        error = h - mu
        v_next = alpha * v - beta * error
        h_next = h + dt * gate * v_next

        # Store outputs
        tl.store(h_out_ptr + offsets, h_next, mask=mask)
        tl.store(v_out_ptr + offsets, v_next, mask=mask)

    class TritonINLDynamicsFunction(torch.autograd.Function):
        """Autograd function wrapping Triton kernels."""

        @staticmethod
        def forward(
            ctx,
            h: torch.Tensor,
            v: torch.Tensor,
            mu: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor,
            gate: torch.Tensor,
            dt: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass using Triton kernel."""
            orig_shape = h.shape
            hidden_size = h.shape[-1]

            # Flatten everything
            h_flat = h.contiguous().view(-1)
            v_flat = v.contiguous().view(-1)
            alpha_flat = alpha.contiguous().view(-1)
            beta_flat = beta.contiguous().view(-1)
            gate_flat = gate.contiguous().view(-1)

            n_total = h_flat.numel()

            # Allocate outputs
            h_out = torch.empty_like(h_flat)
            v_out = torch.empty_like(v_flat)

            # Launch kernel - one block per 1024 elements
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_total, BLOCK_SIZE),)

            _inl_dynamics_kernel[grid](
                h_flat, v_flat, mu,
                alpha_flat, beta_flat, gate_flat,
                h_out, v_out,
                n_total,
                hidden_size,
                dt,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            # Reshape back
            h_next = h_out.view(orig_shape)
            v_next = v_out.view(orig_shape)

            ctx.save_for_backward(h, v, mu, alpha, beta, gate)
            ctx.dt = dt

            return h_next, v_next

        @staticmethod
        def backward(ctx, dh_next: torch.Tensor, dv_next: torch.Tensor):
            """Backward pass using PyTorch (simpler, still fast with compile)."""
            h, v, mu, alpha, beta, gate = ctx.saved_tensors
            dt = ctx.dt

            # Recompute forward
            error = h - mu
            v_next = alpha * v - beta * error

            # Backward through h_next = h + dt * gate * v_next
            dh = dh_next.clone()
            dgate = dh_next * dt * v_next
            dv_next_from_h = dh_next * dt * gate

            # Total dv_next gradient
            dv_next_total = dv_next + dv_next_from_h

            # Backward through v_next = alpha * v - beta * error
            dalpha = dv_next_total * v
            dv = dv_next_total * alpha
            dbeta = -dv_next_total * error
            derror = -dv_next_total * beta

            # Backward through error = h - mu
            dh = dh + derror
            dmu = -derror.sum(dim=list(range(len(derror.shape) - 1)))

            return dh, dv, dmu, dalpha, dbeta, dgate, None

    def triton_inl_dynamics(
        h: torch.Tensor,
        v: torch.Tensor,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        dt: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Triton-accelerated INL dynamics."""
        return TritonINLDynamicsFunction.apply(h, v, mu, alpha, beta, gate, dt)

else:
    # No Triton - use PyTorch
    triton_inl_dynamics = pytorch_inl_dynamics


# Auto-select best implementation
def inl_dynamics(
    h: torch.Tensor,
    v: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    dt: float = 0.1,
    use_triton: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    INL dynamics with automatic backend selection.
    Uses Triton on CUDA if available, falls back to PyTorch.
    """
    if use_triton is None:
        use_triton = HAS_TRITON and h.is_cuda

    if use_triton and HAS_TRITON:
        return triton_inl_dynamics(h, v, mu, alpha, beta, gate, dt)
    else:
        return pytorch_inl_dynamics(h, v, mu, alpha, beta, gate, dt)
