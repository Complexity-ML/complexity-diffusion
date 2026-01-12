"""
INL Dynamics - Robotics-grade control with velocity tracking.

Full dynamics equations (like a physical system):
    error = h - mu                      # deviation from equilibrium
    v_next = alpha * v - beta * error   # velocity update (momentum + correction)
    h_next = h + dt * gate * v_next     # position update (integration)

This creates smooth, stable trajectories like a robot controller.
Supports Triton acceleration when available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import Triton-accelerated version
try:
    from ..cuda.triton_dynamics import inl_dynamics as triton_dynamics_fn
    HAS_TRITON_DYNAMICS = True
except ImportError:
    HAS_TRITON_DYNAMICS = False


class INLDynamics(nn.Module):
    """
    Full INL Dynamics for diffusion transformers.

    Benefits:
        - Smooth denoising trajectories
        - Stable convergence (PID-like control)
        - Learnable dynamics per dimension
        - Real-time capable
        - Triton-accelerated when available
    """

    def __init__(
        self,
        hidden_size: int,
        init_alpha: float = 0.9,      # high inertia = smooth
        init_beta: float = 0.1,       # low correction = stable
        init_gate: float = 0.5,       # medium amplitude
        dt: float = 0.1,              # integration timestep
        controller_hidden: int = 64,  # controller MLP size
        use_triton: bool = True,      # use Triton when available
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.use_triton = use_triton and HAS_TRITON_DYNAMICS

        # Learnable equilibrium (target position)
        self.mu = nn.Parameter(torch.zeros(hidden_size))

        # Controller MLP - computes alpha, beta, gate from context
        # Input: [h, v] concatenated
        self.controller = nn.Sequential(
            nn.Linear(hidden_size * 2, controller_hidden),
            nn.SiLU(),
            nn.Linear(controller_hidden, hidden_size * 3),  # outputs: alpha, beta, gate
        )

        # Initialize controller biases for desired initial values
        with torch.no_grad():
            bias = self.controller[-1].bias
            # alpha in [0,1] via sigmoid, init to ~0.9
            bias[:hidden_size].fill_(2.2)  # sigmoid(2.2) ~ 0.9
            # beta in [0,inf) via softplus, init to ~0.1
            bias[hidden_size:hidden_size*2].fill_(-2.2)  # softplus(-2.2) ~ 0.1
            # gate in [0,1] via sigmoid, init to ~0.5
            bias[hidden_size*2:].fill_(0.0)  # sigmoid(0) = 0.5

            # Small weights for stable start
            self.controller[-1].weight.normal_(0, 0.01)

    def forward(
        self,
        h: torch.Tensor,
        v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply dynamics update.

        Args:
            h: Hidden states [batch, seq_len, hidden_size]
            v: Velocity states [batch, seq_len, hidden_size] (None = init to zero)

        Returns:
            h_next: Updated hidden states
            v_next: Updated velocity states
        """
        # Initialize velocity if not provided
        if v is None:
            v = torch.zeros_like(h)

        # Controller computes adaptive parameters from [h, v]
        controller_input = torch.cat([h, v], dim=-1)
        controller_out = self.controller(controller_input)

        # Split and apply activations
        alpha_raw, beta_raw, gate_raw = torch.split(
            controller_out, self.hidden_size, dim=-1
        )
        alpha = torch.sigmoid(alpha_raw)      # [0, 1] - inertia
        # CRITICAL FIX: Clamp beta to prevent explosion!
        # softplus can go to infinity, causing NaN after long training
        # Max beta=2.0 keeps dynamics stable (like a real PID controller)
        beta = torch.clamp(F.softplus(beta_raw), max=2.0)  # [0, 2] - correction
        gate = torch.sigmoid(gate_raw)        # [0, 1] - amplitude

        # Use Triton-accelerated dynamics if available
        if self.use_triton and h.is_cuda:
            h_next, v_next = triton_dynamics_fn(
                h, v, self.mu, alpha, beta, gate, self.dt
            )
            # STABILITY: Clamp velocity to prevent runaway accumulation
            v_next = torch.clamp(v_next, min=-10.0, max=10.0)
        else:
            # PyTorch fallback
            error = h - self.mu                           # deviation from equilibrium
            v_next = alpha * v - beta * error             # velocity update
            # STABILITY: Clamp velocity to prevent runaway accumulation
            v_next = torch.clamp(v_next, min=-10.0, max=10.0)
            h_next = h + self.dt * gate * v_next          # position update

        return h_next, v_next

    def init_velocity(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Initialize velocity to zero."""
        return torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
