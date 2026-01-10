"""
Triton-accelerated Flash Attention for Complexity DiT.

Based on Flash Attention algorithm with QK-Norm support.
"""

import torch
import math
from typing import Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _flash_attn_fwd_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Flash attention forward kernel."""
        # Program IDs
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        # Batch and head indices
        batch = pid_bh // tl.cdiv(N_CTX, 1)  # placeholder for num_heads
        head = pid_bh % tl.cdiv(N_CTX, 1)

        # Offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, HEAD_DIM)

        # Initialize output accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Load Q block
        q_ptrs = Q + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        # Scale
        qk_scale = 1.0 / tl.sqrt(HEAD_DIM.to(tl.float32))

        # Iterate over K, V blocks
        for start_n in range(0, N_CTX, BLOCK_N):
            curr_offs_n = start_n + offs_n

            # Load K block
            k_ptrs = K + pid_bh * stride_kh + curr_offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=curr_offs_n[:, None] < N_CTX, other=0.0)

            # Load V block
            v_ptrs = V + pid_bh * stride_vh + curr_offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=curr_offs_n[:, None] < N_CTX, other=0.0)

            # Compute attention scores
            qk = tl.dot(q, tl.trans(k)) * qk_scale

            # Mask out-of-bounds
            qk = tl.where(
                (offs_m[:, None] < N_CTX) & (curr_offs_n[None, :] < N_CTX),
                qk,
                float("-inf"),
            )

            # Online softmax update
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            l_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)

            # Update accumulator
            p = tl.exp(qk - m_new[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

            # Update running max and sum
            m_i = m_new
            l_i = l_new

        # Final normalization
        acc = acc / l_i[:, None]

        # Store output
        out_ptrs = Out + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Triton Flash Attention.

    Args:
        q: Query [batch, heads, seq, dim]
        k: Key [batch, heads, seq, dim]
        v: Value [batch, heads, seq, dim]
        scale: Optional scaling factor

    Returns:
        Output [batch, heads, seq, dim]
    """
    batch, heads, seq_len, head_dim = q.shape

    # Flatten batch and heads
    q_flat = q.reshape(batch * heads, seq_len, head_dim).contiguous()
    k_flat = k.reshape(batch * heads, seq_len, head_dim).contiguous()
    v_flat = v.reshape(batch * heads, seq_len, head_dim).contiguous()
    out = torch.empty_like(q_flat)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64

    # Grid
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    # Get strides
    stride_qb, stride_qm, stride_qk = q_flat.stride()
    stride_kb, stride_kn, stride_kk = k_flat.stride()
    stride_vb, stride_vn, stride_vk = v_flat.stride()
    stride_ob, stride_om, stride_ok = out.stride()

    _flash_attn_fwd_kernel[grid](
        q_flat, k_flat, v_flat, out,
        stride_qb, 1, stride_qm, stride_qk,
        stride_kb, 1, stride_kn, stride_kk,
        stride_vb, 1, stride_vn, stride_vk,
        stride_ob, 1, stride_om, stride_ok,
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out.reshape(batch, heads, seq_len, head_dim)


def pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure PyTorch attention (fallback)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    use_triton: Optional[bool] = None,
) -> torch.Tensor:
    """
    Attention with automatic backend selection.

    Prefers PyTorch's native Flash Attention if available,
    falls back to Triton implementation, then pure PyTorch.
    """
    # Try PyTorch native SDPA first (best option)
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            scale=scale,
            is_causal=False,
        )

    if use_triton is None:
        use_triton = HAS_TRITON and q.is_cuda

    if use_triton and HAS_TRITON:
        return triton_flash_attention(q, k, v, scale)
    else:
        return pytorch_attention(q, k, v, scale)
