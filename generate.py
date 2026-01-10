"""
Generate images with trained Complexity DiT.

Usage:
    python generate.py --checkpoint checkpoints/dit/model_final.pt --output samples/
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image

from complexity_diffusion import ComplexityDiT, ComplexityVAE


class DDPMSampler:
    """DDPM sampler for generation."""

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_steps = num_steps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()
        self.sqrt_recip_alphas = (1 / self.alphas).sqrt()

        # Posterior variance
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    @torch.no_grad()
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """Single denoising step."""
        t = timestep

        # Predict x_0
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        beta = self.betas[t]

        # Mean of posterior
        pred_original = sqrt_recip_alpha * (sample - beta / sqrt_one_minus_alpha_cumprod * model_output)

        # Variance
        if t > 0:
            noise = torch.randn_like(sample)
            variance = self.posterior_variance[t].sqrt()
            sample = pred_original + variance * noise
        else:
            sample = pred_original

        return sample


@torch.no_grad()
def generate(
    model: ComplexityDiT,
    vae: ComplexityVAE,
    sampler: DDPMSampler,
    num_samples: int = 4,
    num_steps: int = 50,
    cfg_scale: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate images."""
    model.eval()

    # Start from noise
    latents = torch.randn(num_samples, 4, 32, 32, device=device)

    # Dummy context (unconditional)
    context = torch.zeros(num_samples, 77, model.d_model, device=device)

    # DDIM-like sampling with fewer steps
    step_ratio = sampler.num_steps // num_steps
    timesteps = list(range(0, sampler.num_steps, step_ratio))[::-1]

    for t in tqdm(timesteps, desc="Sampling"):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)

        if cfg_scale > 1.0:
            noise_pred = model.forward_with_cfg(latents, t_tensor, context, cfg_scale)
        else:
            noise_pred = model(latents, t_tensor, context)

        latents = sampler.step(noise_pred, t, latents)

    # Decode
    images = vae.decode(latents)
    images = (images + 1) / 2  # [-1, 1] -> [0, 1]
    images = images.clamp(0, 1)

    return images


def save_images(images: torch.Tensor, output_dir: Path, prefix: str = "sample"):
    """Save generated images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        Image.fromarray(img_np).save(output_dir / f"{prefix}_{i:04d}.png")

    print(f"Saved {len(images)} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate with Complexity DiT')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vae_path', type=str, default=None,
                        help='Path to VAE weights')
    parser.add_argument('--output', type=str, default='samples/',
                        help='Output directory')

    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Denoising steps (fewer = faster)')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                        help='CFG scale (1.0 = unconditional)')

    parser.add_argument('--config', type=str, default='S',
                        help='Model config')
    parser.add_argument('--context_dim', type=int, default=768)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = ComplexityDiT.from_config(args.config, context_dim=args.context_dim).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from step {checkpoint['step']}")

    # Load VAE
    print("Loading VAE...")
    vae = ComplexityVAE(image_size=256, base_channels=128, latent_dim=4).to(device)
    if args.vae_path:
        from safetensors.torch import load_file
        vae.load_state_dict(load_file(args.vae_path))

    # Sampler
    sampler = DDPMSampler(num_steps=1000).to(device)

    # Generate
    print(f"\nGenerating {args.num_samples} images...")
    images = generate(
        model, vae, sampler,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        device=device,
    )

    # Save
    save_images(images, Path(args.output))


if __name__ == '__main__':
    main()
