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


if HAS_TRITON:
    @triton.jit
    def _inl_dynamics_fwd_kernel(
        # Pointers
        h_ptr, v_ptr, mu_ptr,
        alpha_ptr, beta_ptr, gate_ptr,
        h_out_ptr, v_out_ptr,
        # Scalars
        dt: tl.constexpr,
        hidden_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Forward pass of INL dynamics."""
        # Program ID
        pid = tl.program_id(0)

        # Block start
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size

        # Load inputs
        h = tl.load(h_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
        mu = tl.load(mu_ptr + offsets, mask=mask, other=0.0)
        alpha = tl.load(alpha_ptr + offsets, mask=mask, other=0.0)
        beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)

        # Dynamics computation (fused)
        error = h - mu
        v_next = alpha * v - beta * error
        h_next = h + dt * gate * v_next

        # Store outputs
        tl.store(h_out_ptr + offsets, h_next, mask=mask)
        tl.store(v_out_ptr + offsets, v_next, mask=mask)


    @triton.jit
    def _inl_dynamics_bwd_kernel(
        # Forward inputs
        h_ptr, v_ptr, mu_ptr,
        alpha_ptr, beta_ptr, gate_ptr,
        # Gradients from output
        dh_next_ptr, dv_next_ptr,
        # Gradients to compute
        dh_ptr, dv_ptr, dmu_ptr,
        dalpha_ptr, dbeta_ptr, dgate_ptr,
        # Scalars
        dt: tl.constexpr,
        hidden_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Backward pass of INL dynamics."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size

        # Load forward values
        h = tl.load(h_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
        mu = tl.load(mu_ptr + offsets, mask=mask, other=0.0)
        alpha = tl.load(alpha_ptr + offsets, mask=mask, other=0.0)
        beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)

        # Load output gradients
        dh_next = tl.load(dh_next_ptr + offsets, mask=mask, other=0.0)
        dv_next = tl.load(dv_next_ptr + offsets, mask=mask, other=0.0)

        # Recompute forward intermediates
        error = h - mu
        v_next = alpha * v - beta * error

        # Backward through h_next = h + dt * gate * v_next
        dh = dh_next  # gradient flows through identity
        dgate = dh_next * dt * v_next
        dv_next_from_h = dh_next * dt * gate

        # Total dv_next
        dv_next_total = dv_next + dv_next_from_h

        # Backward through v_next = alpha * v - beta * error
        dalpha = dv_next_total * v
        dv = dv_next_total * alpha
        dbeta = -dv_next_total * error
        derror = -dv_next_total * beta

        # Backward through error = h - mu
        dh = dh + derror
        dmu = -derror

        # Store gradients
        tl.store(dh_ptr + offsets, dh, mask=mask)
        tl.store(dv_ptr + offsets, dv, mask=mask)
        tl.store(dmu_ptr + offsets, dmu, mask=mask)
        tl.store(dalpha_ptr + offsets, dalpha, mask=mask)
        tl.store(dbeta_ptr + offsets, dbeta, mask=mask)
        tl.store(dgate_ptr + offsets, dgate, mask=mask)


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
        # Flatten for kernel
        batch_seq = h.shape[:-1]
        hidden_size = h.shape[-1]

        h_flat = h.reshape(-1, hidden_size).contiguous()
        v_flat = v.reshape(-1, hidden_size).contiguous()
        alpha_flat = alpha.reshape(-1, hidden_size).contiguous()
        beta_flat = beta.reshape(-1, hidden_size).contiguous()
        gate_flat = gate.reshape(-1, hidden_size).contiguous()

        # Outputs
        h_out = torch.empty_like(h_flat)
        v_out = torch.empty_like(v_flat)

        # Launch kernel for each batch*seq element
        n_elements = h_flat.shape[0]
        BLOCK_SIZE = 128

        for i in range(n_elements):
            grid = (triton.cdiv(hidden_size, BLOCK_SIZE),)
            _inl_dynamics_fwd_kernel[grid](
                h_flat[i], v_flat[i], mu,
                alpha_flat[i], beta_flat[i], gate_flat[i],
                h_out[i], v_out[i],
                dt=dt,
                hidden_size=hidden_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        # Reshape outputs
        h_next = h_out.reshape(*batch_seq, hidden_size)
        v_next = v_out.reshape(*batch_seq, hidden_size)

        # Save for backward
        ctx.save_for_backward(h, v, mu, alpha, beta, gate)
        ctx.dt = dt

        return h_next, v_next

    @staticmethod
    def backward(ctx, dh_next: torch.Tensor, dv_next: torch.Tensor):
        """Backward pass using Triton kernel."""
        h, v, mu, alpha, beta, gate = ctx.saved_tensors
        dt = ctx.dt

        batch_seq = h.shape[:-1]
        hidden_size = h.shape[-1]

        # Flatten
        h_flat = h.reshape(-1, hidden_size).contiguous()
        v_flat = v.reshape(-1, hidden_size).contiguous()
        alpha_flat = alpha.reshape(-1, hidden_size).contiguous()
        beta_flat = beta.reshape(-1, hidden_size).contiguous()
        gate_flat = gate.reshape(-1, hidden_size).contiguous()
        dh_next_flat = dh_next.reshape(-1, hidden_size).contiguous()
        dv_next_flat = dv_next.reshape(-1, hidden_size).contiguous()

        # Gradient outputs
        dh = torch.empty_like(h_flat)
        dv = torch.empty_like(v_flat)
        dmu_accum = torch.zeros(hidden_size, device=h.device, dtype=h.dtype)
        dalpha = torch.empty_like(alpha_flat)
        dbeta = torch.empty_like(beta_flat)
        dgate = torch.empty_like(gate_flat)

        n_elements = h_flat.shape[0]
        BLOCK_SIZE = 128

        for i in range(n_elements):
            dmu_i = torch.empty(hidden_size, device=h.device, dtype=h.dtype)
            grid = (triton.cdiv(hidden_size, BLOCK_SIZE),)
            _inl_dynamics_bwd_kernel[grid](
                h_flat[i], v_flat[i], mu,
                alpha_flat[i], beta_flat[i], gate_flat[i],
                dh_next_flat[i], dv_next_flat[i],
                dh[i], dv[i], dmu_i,
                dalpha[i], dbeta[i], dgate[i],
                dt=dt,
                hidden_size=hidden_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            dmu_accum += dmu_i

        # Reshape
        dh = dh.reshape(*batch_seq, hidden_size)
        dv = dv.reshape(*batch_seq, hidden_size)
        dalpha = dalpha.reshape(*batch_seq, hidden_size)
        dbeta = dbeta.reshape(*batch_seq, hidden_size)
        dgate = dgate.reshape(*batch_seq, hidden_size)

        return dh, dv, dmu_accum, dalpha, dbeta, dgate, None


def triton_inl_dynamics(
    h: torch.Tensor,
    v: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    dt: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated INL dynamics.

    Args:
        h: Hidden states [batch, seq, hidden]
        v: Velocity states [batch, seq, hidden]
        mu: Equilibrium position [hidden]
        alpha: Inertia parameter [batch, seq, hidden]
        beta: Correction parameter [batch, seq, hidden]
        gate: Amplitude parameter [batch, seq, hidden]
        dt: Integration timestep

    Returns:
        h_next, v_next: Updated states
    """
    return TritonINLDynamicsFunction.apply(h, v, mu, alpha, beta, gate, dt)


# Fallback PyTorch implementation
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
