# Complexity Diffusion

Diffusion Transformer (DiT) with **INL Dynamics** for image generation.

[![PyPI version](https://badge.fury.io/py/complexity-diffusion.svg)](https://badge.fury.io/py/complexity-diffusion)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install complexity-diffusion
```

## Features

- **ComplexityDiT** - Diffusion Transformer with INL Dynamics
- **ComplexityVAE** - Image encoder/decoder
- **DDIMScheduler** - Fast sampling (50 steps vs 1000)
- **Unconditional generation** (text-to-image coming soon)

## Architecture

Each DiT block has 4 components:

1. **KQV Attention** with QK-Norm (self-attention)
2. **Cross-Attention** for text conditioning
3. **INL Dynamics** (smooth denoising trajectories)
4. **Token-Routed MLP** with experts

### INL Dynamics for Diffusion

The Dynamics layer provides smooth denoising:

```python
error = h - mu                      # deviation from equilibrium
v_next = alpha * v - beta * error   # velocity update
h_next = h + dt * gate * v_next     # position update
```

This creates smooth, stable denoising trajectories.

## Usage

```python
from complexity_diffusion import ComplexityDiT, ComplexityVAE

# Create DiT model
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

| Config | Params | Layers | d_model | Heads |
|--------|--------|--------|---------|-------|
| S      | ~114M  | 12     | 384     | 6     |
| B      | ~250M  | 12     | 768     | 12    |
| L      | ~500M  | 24     | 1024    | 16    |
| XL     | ~700M  | 28     | 1152    | 16    |
| XXL    | ~1.5B  | 32     | 1536    | 24    |

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

# Generate
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
    --steps 50000
```

## Related Packages

- **complexity** - LLM architecture (Token-Routed MLP)
- **complexity-deep** - LLM with INL Dynamics
- **pyllm-inference** - Inference server

## License

MIT
