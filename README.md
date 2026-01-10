# Complexity Diffusion

Llama-style Diffusion Transformer with INL Dynamics for image generation.

## Architecture

Multicouche robotics architecture per transformer block:

1. **KQV Attention** with QK-Norm (perception)
2. **Cross-Attention** for text conditioning
3. **INL Dynamics** (control with velocity tracking)
4. **Token-Routed MLP** with experts (transformation)

## Features

- Full velocity tracking (smooth denoising trajectories)
- GQA (Grouped Query Attention)
- QK Normalization (stable training)
- Token-Routed MLP with experts
- AdaLN-Zero conditioning
- RoPE 2D positional encoding

## Installation

```bash
pip install complexity-diffusion
```

## Usage

```python
from complexity_diffusion import ComplexityDiT, ComplexityVAE

# Create DiT model
dit = ComplexityDiT.from_config("L")  # Large ~500M params
print(f"Parameters: {dit.get_num_params() / 1e6:.1f}M")

# Create VAE
vae = ComplexityVAE()

# Forward pass
import torch
x = torch.randn(1, 4, 32, 32)  # Latent
t = torch.tensor([500])  # Timestep
context = torch.randn(1, 77, 2048)  # Text embeddings

noise_pred = dit(x, t, context)
```

## Model Configurations

| Config | Params | Layers | d_model | Heads |
|--------|--------|--------|---------|-------|
| S      | ~100M  | 12     | 384     | 6     |
| B      | ~250M  | 12     | 768     | 12    |
| L      | ~500M  | 24     | 1024    | 16    |
| XL     | ~700M  | 28     | 1152    | 16    |
| XXL    | ~1.5B  | 32     | 1536    | 24    |

## INL Dynamics

The INL Dynamics module provides robotics-grade control:

```
error = h - mu                      # deviation from equilibrium
v_next = alpha * v - beta * error   # velocity update
h_next = h + dt * gate * v_next     # position update
```

This creates smooth, stable denoising trajectories.

## License

MIT
