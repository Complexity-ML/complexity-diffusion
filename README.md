# Complexity Diffusion v0.3.0

Diffusion Transformer (DiT) with **Mu-Guided Architecture** and **INL Dynamics** for image generation.

[![PyPI version](https://badge.fury.io/py/complexity-diffusion.svg)](https://badge.fury.io/py/complexity-diffusion)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install complexity-diffusion
```

## What's New in v0.3.0

- **Mu-Guided Attention (KQV)**: μ biases K, Q, AND V for smoother denoising
- **Mu-Guided Expert Routing**: μ influences which expert processes each patch
- **Contextual Mu**: μ adapts based on denoising progress
- **Mu-Damped Dynamics**: Top-down guidance reduces oscillations
- **Fused Mu-KQV**: 2x faster via concat+cuBLAS

## Features

- **ComplexityDiT** - Diffusion Transformer with Mu-Guided INL Dynamics
- **ComplexityVAE** - Image encoder/decoder
- **DDIMScheduler** - Fast sampling (50 steps vs 1000)
- **Unconditional generation** (text-to-image coming soon)

## Architecture (v0.3.0)

Each DiT block has 4 components with **Mu guidance**:

```
Noisy Latent + Timestep
  │
  ▼
[AdaLN] ─► [Mu-Guided KQV Attention] ─► [INL Dynamics] ─► [Cross-Attention] ─► [Token-Routed MLP]
  │              ▲                            │                                      ▲
  │              │                            │                                      │
  │         mu_prev                     mu_contextual ───────────────────────────────┘
  │                                           │
  +────────────────── Residual ───────────────┼───────────────────────────────────────+
  │                                           │
  ▼                                           ▼
Output ◄────────────────────────────── mu_next (to next block)
```

### Mu-Guided Diffusion

```python
# Mu guides the denoising process
x_mu = concat([x, mu_prev], dim=-1)
k = x_mu @ concat([W_k, W_mu_k])  # K biased by mu
q = x_mu @ concat([W_q, W_mu_q])  # Q biased by mu
v = x_mu @ concat([W_v, W_mu_v])  # V biased by mu

# Mu-Damped dynamics for smooth denoising
error = h - mu
v_next = alpha * v - beta * error
h_next = h + dt * gate * v_next
mu_contextual = mu + mu_proj(h)  # Adapts to denoising state
```

### Why Mu for Diffusion?

| Aspect | Standard DiT | Mu-Guided DiT |
|--------|--------------|---------------|
| **Denoising** | Each step independent | μ guides trajectory |
| **Oscillations** | Can be jerky | **Smooth (damped)** |
| **Convergence** | Slow | **Fast** |
| **Sample quality** | Good | **Better (smoother)** |

## Usage

```python
from complexity_diffusion import ComplexityDiT, ComplexityVAE

# Create DiT model (v0.3.0 with Mu-Guided)
dit = ComplexityDiT.from_config("S")  # Small ~114M params
print(f"Parameters: {dit.get_num_params() / 1e6:.1f}M")

# Create VAE
vae = ComplexityVAE(image_size=256, latent_dim=4)

# Forward pass
import torch
x = torch.randn(1, 4, 32, 32)  # Latent [B, C, H, W]
t = torch.tensor([500])  # Timestep
context = torch.randn(1, 77, 768)  # Text embeddings (optional)

noise_pred = dit(x, t, context)
```

## Model Configurations

| Config | Params | Layers | d_model | Heads | Experts |
|--------|--------|--------|---------|-------|---------|
| S      | ~114M  | 12     | 384     | 6     | 4       |
| B      | ~250M  | 12     | 768     | 12    | 4       |
| L      | ~500M  | 24     | 1024    | 16    | 8       |
| XL     | ~700M  | 28     | 1152    | 16    | 8       |
| XXL    | ~1.5B  | 32     | 1536    | 24    | 8       |

## Generation

```python
from complexity_diffusion import ComplexityDiT, ComplexityVAE
from complexity_diffusion.pipeline.text_to_image import DDIMScheduler
import torch

device = "cuda"

# Load models
dit = ComplexityDiT.from_config("S").to(device)
vae = ComplexityVAE().to(device)
scheduler = DDIMScheduler(num_train_timesteps=1000)

# Generate with Mu-guided denoising
scheduler.set_timesteps(50)  # 50 sampling steps
latents = torch.randn(1, 4, 32, 32, device=device)
context = torch.zeros(1, 77, 768, device=device)  # Unconditional

for t in scheduler.timesteps:
    noise_pred = dit(latents, t.unsqueeze(0), context)
    latents = scheduler.step(noise_pred, t.item(), latents)

# Decode to image
images = vae.decode(latents)  # [B, 3, 256, 256]
```

## Pre-trained Models

| Model | Dataset | Steps | HuggingFace |
|-------|---------|-------|-------------|
| ComplexityDiT-S | WikiArt | 20K | [Pacific-Prime/diffusion-DiT](https://huggingface.co/Pacific-Prime/diffusion-DiT) |

```python
# Load from HuggingFace
from safetensors.torch import load_file

state_dict = load_file("model.safetensors")
dit = ComplexityDiT.from_config("S", context_dim=768)
dit.load_state_dict(state_dict)
```

## Training

```bash
# Train DiT on WikiArt
python train_dit.py \
    --dataset huggan/wikiart \
    --dit-size S \
    --batch-size 32 \
    --steps 50000 \
    --bf16
```

## Innovations

| Innovation | Status | Description |
|------------|--------|-------------|
| Mu-Guided KQV | **Novel** | μ biases K, Q, V for coherent denoising |
| Mu-Guided Experts | **Novel** | μ influences patch routing |
| Mu-Damped Dynamics | **Novel** | Top-down damping for smooth trajectories |
| Contextual Mu | **Novel** | μ adapts to denoising progress |
| Token-Routed MLP | **Novel** | Deterministic patch routing |

## Related Packages

- [complexity-deep](https://github.com/Complexity-ML/complexity-deep) - LLM with Mu-Guided architecture
- [complexity-framework](https://github.com/Complexity-ML/complexity-framework) - Training framework
- [pacific-prime](https://huggingface.co/Pacific-Prime/pacific-prime) - 1.5B LLM checkpoint

## License

MIT
