"""
Complexity DiT - Diffusion Transformer with Llama/Complexity Architecture.

Multicouche architecture per layer:
    1. KQV Attention with QK-Norm (perception)
    2. INL Dynamics (control with velocity tracking)
    3. Token-Routed MLP (transformation)

Features:
- Full INL Dynamics (robotics-grade control)
- GQA (Grouped Query Attention)
- RoPE 2D positional encoding
- QK Normalization for stability
- AdaLN-Zero conditioning
- Token-Routed MLP with experts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import math

from complexity_diffusion.core.normalization import RMSNorm
from complexity_diffusion.core.dynamics import INLDynamics


@dataclass
class ComplexityDiTConfig:
    """Configuration for Complexity DiT."""
    # Image/Latent
    img_size: int = 32
    patch_size: int = 2
    in_channels: int = 4

    # Architecture
    d_model: int = 1152
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 4
    mlp_ratio: float = 4.0

    # Text conditioning
    context_dim: int = 2048

    # Dropout
    dropout: float = 0.0

    # Token-Routed MLP
    use_token_routed_mlp: bool = True
    num_experts: int = 4

    # INL Dynamics
    dynamics_alpha: float = 0.9
    dynamics_beta: float = 0.1
    dynamics_gate: float = 0.5
    dynamics_dt: float = 0.1
    dynamics_controller_hidden: int = 64

    # 2024 Innovations
    use_qk_norm: bool = True

    # CFG
    use_learned_null: bool = True


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding for image patches."""

    def __init__(self, dim: int, max_size: int = 64):
        super().__init__()
        self.dim = dim
        self.max_size = max_size

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rotary embeddings for h x w grid."""
        y_pos = torch.arange(h, device=device).float()
        x_pos = torch.arange(w, device=device).float()

        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing="ij")
        y_grid = y_grid.flatten()
        x_grid = x_grid.flatten()

        y_emb = torch.outer(y_grid, self.inv_freq)
        x_emb = torch.outer(x_grid, self.inv_freq)

        emb = torch.cat([y_emb, x_emb], dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to queries and keys."""
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_rot, k_rot


class ComplexityAttention(nn.Module):
    """
    Mu-Guided KQV Attention with GQA and QK-Norm (v0.3.0).

    v0.3.0: KQV order (industry standard)
    v0.3.0: Mu-to-KQV projections (INL 2025)
    v0.3.0: Fused Mu-KQV via concat+cuBLAS (2x faster)
    """

    # Flag for fused mu-kqv concat (2x faster than separate matmuls)
    USE_FUSED_MU_KQV = True

    def __init__(
        self,
        d_model: int,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_dim = self.head_dim * num_kv_heads
        self.use_qk_norm = use_qk_norm

        # v0.3.0: KQV order (industry standard like Llama, Qwen, GPT)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # v0.3.0: Mu-to-KQV projections (INL 2025)
        self.mu_to_k = nn.Linear(d_model, self.kv_dim, bias=False)
        self.mu_to_q = nn.Linear(d_model, d_model, bias=False)
        self.mu_to_v = nn.Linear(d_model, self.kv_dim, bias=False)
        # Initialize to zero (start without mu influence, learn it)
        nn.init.zeros_(self.mu_to_k.weight)
        nn.init.zeros_(self.mu_to_q.weight)
        nn.init.zeros_(self.mu_to_v.weight)

        # QK Normalization (2024 innovation)
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        # v0.3.0: Fused Mu-KQV via concat+cuBLAS (2x faster)
        if self.USE_FUSED_MU_KQV and mu_prev is not None:
            # Concat x and mu, then single matmul with concatenated weights
            x_mu = torch.cat([x, mu_prev], dim=-1)  # [B, N, 2*D]

            # Fused KQV projection
            wk = torch.cat([self.k_proj.weight, self.mu_to_k.weight], dim=1)
            wq = torch.cat([self.q_proj.weight, self.mu_to_q.weight], dim=1)
            wv = torch.cat([self.v_proj.weight, self.mu_to_v.weight], dim=1)

            k = F.linear(x_mu, wk).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = F.linear(x_mu, wq).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = F.linear(x_mu, wv).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        elif mu_prev is not None:
            # Separate projections (slower but more readable)
            k = (self.k_proj(x) + self.mu_to_k(mu_prev)).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = (self.q_proj(x) + self.mu_to_q(mu_prev)).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = (self.v_proj(x) + self.mu_to_v(mu_prev)).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            # No mu guidance
            k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QK-Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply rotary embeddings
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads for GQA
        num_rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(num_rep, dim=1)
        v = v.repeat_interleave(num_rep, dim=1)

        # Attention (use SDPA if available)
        try:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        except:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class ComplexityCrossAttention(nn.Module):
    """Cross-attention for text conditioning with QK-Norm."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        num_heads: int = 16,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_qk_norm = use_qk_norm

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(context_dim, d_model, bias=False)
        self.v_proj = nn.Linear(context_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        _, S, _ = context.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        try:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        except:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class TokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP with Mu-Guided routing (v0.3.0).

    v0.3.0: Mu-guided expert routing (INL 2025)

    Routes different patch types to specialized experts,
    with mu providing contextual guidance.
    """

    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        hidden_dim = int(d_model * mlp_ratio)

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # v0.3.0: Mu-guided expert routing (INL 2025)
        self.mu_router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.zeros_(self.mu_router.weight)  # Start neutral

        # Expert MLPs (SwiGLU style)
        self.experts_gate = nn.ModuleList([
            nn.Linear(d_model, hidden_dim, bias=False)
            for _ in range(num_experts)
        ])
        self.experts_up = nn.ModuleList([
            nn.Linear(d_model, hidden_dim, bias=False)
            for _ in range(num_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(hidden_dim, d_model, bias=False)
            for _ in range(num_experts)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mu_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape

        # v0.3.0: Mu-guided routing
        router_logits = self.router(x)  # [B, N, num_experts]
        if mu_prev is not None:
            router_logits = router_logits + self.mu_router(mu_prev)
        router_weights = F.softmax(router_logits, dim=-1)

        # Compute all expert outputs
        expert_outputs = []
        for i in range(self.num_experts):
            gate = F.silu(self.experts_gate[i](x))
            up = self.experts_up[i](x)
            expert_out = self.experts_down[i](gate * up)
            expert_outputs.append(expert_out)

        # Stack and weight by router
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, N, D, num_experts]
        output = torch.einsum("bnde,bne->bnd", expert_outputs, router_weights)

        return self.dropout(output)


class ComplexityMLP(nn.Module):
    """Standard SwiGLU MLP (when not using Token-Routed)."""

    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class ComplexityDiTBlock(nn.Module):
    """
    Complexity DiT Block - Multicouche robotics architecture.

    Architecture (3 components):
        1. KQV Attention with QK-Norm (perception)
        2. Cross-Attention for text (context)
        3. INL Dynamics (control with velocity)
        4. Token-Routed MLP (transformation)

    + AdaLN-Zero for timestep conditioning
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        context_dim: int = 2048,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_token_routed_mlp: bool = True,
        num_experts: int = 4,
        use_qk_norm: bool = True,
        # INL Dynamics
        dynamics_alpha: float = 0.9,
        dynamics_beta: float = 0.1,
        dynamics_gate: float = 0.5,
        dynamics_dt: float = 0.1,
        dynamics_controller_hidden: int = 64,
    ):
        super().__init__()

        # Normalization layers
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.norm_context = RMSNorm(context_dim)

        # 1. KQV Attention (perception)
        self.self_attn = ComplexityAttention(
            d_model, num_heads, num_kv_heads, dropout, use_qk_norm
        )

        # 2. Cross-attention (context)
        self.cross_attn = ComplexityCrossAttention(
            d_model, context_dim, num_heads, dropout, use_qk_norm
        )

        # 3. INL Dynamics (control)
        self.dynamics = INLDynamics(
            hidden_size=d_model,
            init_alpha=dynamics_alpha,
            init_beta=dynamics_beta,
            init_gate=dynamics_gate,
            dt=dynamics_dt,
            controller_hidden=dynamics_controller_hidden,
        )

        # 4. Token-Routed MLP (transformation)
        if use_token_routed_mlp:
            self.mlp = TokenRoutedMLP(d_model, mlp_ratio, num_experts, dropout)
        else:
            self.mlp = ComplexityMLP(d_model, mlp_ratio, dropout)

        # AdaLN-Zero modulation (6 params: shift, scale, gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_emb: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass through multicouche block (v0.3.0 with Mu guidance).

        Args:
            x: Patch embeddings [B, N, D]
            context: Text embeddings [B, S, context_dim]
            t_emb: Timestep embedding [B, D]
            velocity: Velocity state [B, N, D]
            mu_prev: Mu from previous layer [B, N, D]
            cos, sin: Rotary position embeddings

        Returns:
            x: Updated embeddings
            velocity: Updated velocity
            mu_contextual: Contextual mu for next layer
            aux: Auxiliary info
        """
        # Get AdaLN modulation parameters
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)

        # === 1. MU-GUIDED KQV ATTENTION (perception) ===
        x_norm = modulate(self.norm1(x), shift_attn, scale_attn)
        x = x + gate_attn.unsqueeze(1) * self.self_attn(x_norm, cos, sin, mu_prev=mu_prev)

        # === 2. CROSS-ATTENTION (context) ===
        x = x + self.cross_attn(self.norm2(x), self.norm_context(context))

        # === 3. INL DYNAMICS (control) with contextual mu ===
        x, velocity, mu_contextual = self.dynamics(x, velocity, return_mu=True)

        # === 4. MU-GUIDED TOKEN-ROUTED MLP (transformation) ===
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm, mu_prev=mu_contextual)

        aux = {"velocity": velocity, "mu": mu_contextual}
        return x, velocity, mu_contextual, aux


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, d_model: int, max_period: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        d_model: int = 1152,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.proj = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FinalLayer(nn.Module):
    """Final layer for noise prediction."""

    def __init__(self, d_model: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.linear = nn.Linear(d_model, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class ComplexityDiT(nn.Module):
    """
    Complexity Diffusion Transformer.

    Llama-style architecture with INL Dynamics for image generation.

    Features:
    - Multicouche blocks (KQV + Dynamics + MLP)
    - Full velocity tracking (robotics-grade)
    - GQA + QK-Norm
    - Token-Routed MLP with experts
    - AdaLN-Zero conditioning
    """

    # Preset configurations
    CONFIGS = {
        "S": {  # Small ~100M
            "d_model": 384,
            "num_layers": 12,
            "num_heads": 6,
            "num_kv_heads": 2,
        },
        "B": {  # Base ~250M
            "d_model": 768,
            "num_layers": 12,
            "num_heads": 12,
            "num_kv_heads": 4,
        },
        "L": {  # Large ~500M
            "d_model": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "num_kv_heads": 4,
        },
        "XL": {  # XL ~700M
            "d_model": 1152,
            "num_layers": 28,
            "num_heads": 16,
            "num_kv_heads": 4,
        },
        "XXL": {  # XXL ~1.5B
            "d_model": 1536,
            "num_layers": 32,
            "num_heads": 24,
            "num_kv_heads": 6,
        },
    }

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        d_model: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        context_dim: int = 2048,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_token_routed_mlp: bool = True,
        num_experts: int = 4,
        use_qk_norm: bool = True,
        use_learned_null: bool = True,
        # INL Dynamics
        dynamics_alpha: float = 0.9,
        dynamics_beta: float = 0.1,
        dynamics_gate: float = 0.5,
        dynamics_dt: float = 0.1,
        dynamics_controller_hidden: int = 64,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.out_channels = in_channels
        self.use_learned_null = use_learned_null

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, d_model)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        # Timestep embedding
        self.t_embed = TimestepEmbedding(d_model)

        # 2D Rotary position embedding
        self.rope = RotaryPositionEmbedding2D(d_model // num_heads)

        # Learned null embedding for CFG
        if use_learned_null:
            self.null_embedding = nn.Parameter(torch.randn(1, 77, context_dim) * 0.02)

        # Transformer blocks with INL Dynamics
        self.blocks = nn.ModuleList([
            ComplexityDiTBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                context_dim=context_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_token_routed_mlp=use_token_routed_mlp,
                num_experts=num_experts,
                use_qk_norm=use_qk_norm,
                dynamics_alpha=dynamics_alpha,
                dynamics_beta=dynamics_beta,
                dynamics_gate=dynamics_gate,
                dynamics_dt=dynamics_dt,
                dynamics_controller_hidden=dynamics_controller_hidden,
            )
            for _ in range(num_layers)
        ])

        # Final layer
        self.final_layer = FinalLayer(d_model, patch_size, self.out_channels)

        # Initialize weights
        self._init_weights()

        print(f"ComplexityDiT initialized: {num_layers}L, d={d_model}, "
              f"experts={num_experts}, dynamics=enabled")

    def _init_weights(self):
        """Initialize weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Zero-out output layers for residual connections
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch predictions back to image."""
        c = self.out_channels
        p = self.patch_size
        h = w = self.grid_size

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))

        return imgs

    def get_null_embedding(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get null embedding for CFG."""
        if self.use_learned_null:
            return self.null_embedding.expand(batch_size, -1, -1)[:, :seq_len]
        else:
            return torch.zeros(batch_size, seq_len, self.d_model, device=device)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        use_null_context: bool = False,
        velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for noise prediction.

        Args:
            x: Noisy latent [B, C, H, W]
            t: Timestep [B]
            context: Text embeddings [B, S, context_dim]
            use_null_context: If True, use null embedding (for CFG)
            velocity: Optional initial velocity

        Returns:
            Predicted noise [B, C, H, W]
        """
        B = x.shape[0]

        # Use null embedding for unconditional generation
        if use_null_context:
            context = self.get_null_embedding(B, context.shape[1], x.device)

        # Patch embed
        x = self.patch_embed(x)

        # Initialize velocity and mu
        if velocity is None:
            velocity = torch.zeros_like(x)
        mu_prev = None  # First layer has no mu guidance

        # Timestep embedding
        t_emb = self.t_embed(t)

        # Get rotary embeddings
        cos, sin = self.rope(self.grid_size, self.grid_size, x.device)

        # v0.3.0: Transformer blocks with velocity tracking and mu guidance
        for block in self.blocks:
            x, velocity, mu_prev, _ = block(x, context, t_emb, velocity, mu_prev, cos, sin)

        # Final layer
        x = self.final_layer(x, t_emb)

        # Unpatchify
        x = self.unpatchify(x)

        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        cfg_scale: float = 7.5,
    ) -> torch.Tensor:
        """Forward pass with Classifier-Free Guidance."""
        noise_uncond = self.forward(x, t, context, use_null_context=True)
        noise_cond = self.forward(x, t, context, use_null_context=False)
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        return noise_pred

    @classmethod
    def from_config(cls, config_name: str, **kwargs) -> "ComplexityDiT":
        """Create model from predefined config."""
        if config_name not in cls.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Choose from {list(cls.CONFIGS.keys())}")

        config = cls.CONFIGS[config_name].copy()
        config.update(kwargs)
        return cls(**config)

    def get_num_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
