"""
Train Complexity DiT - Minimal setup for image generation.

Usage:
    python train_dit.py --config S --batch_size 16 --steps 100000
    python train_dit.py --config B --batch_size 8 --steps 200000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm

# Complexity imports
from complexity_diffusion import ComplexityDiT, ComplexityVAE


# =============================================================================
# DIFFUSION SCHEDULER
# =============================================================================

class DDPMScheduler:
    """Simple DDPM scheduler."""

    def __init__(
        self,
        num_train_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_train_steps = num_train_steps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples."""
        sqrt_alpha_cumprod = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[timesteps]).sqrt()

        # Reshape for broadcasting
        while len(sqrt_alpha_cumprod.shape) < len(x.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self


# =============================================================================
# SIMPLE DATASET
# =============================================================================

class ImageFolderDataset(torch.utils.data.Dataset):
    """Simple image folder dataset."""

    def __init__(self, root: str, image_size: int = 256):
        from torchvision import transforms
        from PIL import Image

        self.root = Path(root)
        self.image_size = image_size

        # Find images
        self.images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.images.extend(self.root.glob(f'**/{ext}'))

        print(f"Found {len(self.images)} images in {root}")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.images[idx]).convert('RGB')
        return self.transform(img)


class LatentDataset(torch.utils.data.Dataset):
    """Pre-encoded latent dataset (faster training)."""

    def __init__(self, latent_dir: str):
        self.latent_dir = Path(latent_dir)
        self.files = list(self.latent_dir.glob('*.pt'))
        print(f"Found {len(self.files)} latent files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])


# =============================================================================
# TRAINING
# =============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    print(f"\nCreating ComplexityDiT-{args.config}...")
    model = ComplexityDiT.from_config(
        args.config,
        context_dim=args.context_dim,
    ).to(device)

    num_params = model.get_num_params() / 1e6
    print(f"Parameters: {num_params:.1f}M")

    # Load VAE
    print("\nLoading VAE...")
    vae = ComplexityVAE(
        image_size=256,
        base_channels=128,
        latent_dim=4,
    ).to(device)

    if args.vae_path:
        from safetensors.torch import load_file
        vae_state = load_file(args.vae_path)
        vae.load_state_dict(vae_state)
        print(f"Loaded VAE from {args.vae_path}")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    if args.use_latents:
        dataset = LatentDataset(args.data_dir)
    else:
        dataset = ImageFolderDataset(args.data_dir, image_size=256)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Scheduler
    scheduler = DDPMScheduler(num_train_steps=1000).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None

    # Training loop
    print(f"\nStarting training for {args.steps} steps...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    model.train()
    global_step = 0
    losses = []

    # Dummy context (unconditional for now)
    dummy_context = torch.zeros(args.batch_size, 77, args.context_dim, device=device)

    pbar = tqdm(total=args.steps, desc="Training")

    while global_step < args.steps:
        for batch in dataloader:
            if global_step >= args.steps:
                break

            # Get latents
            if args.use_latents:
                latents = batch.to(device)
            else:
                images = batch.to(device)
                with torch.no_grad():
                    latents = vae.encode(images, sample=True)

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)

            # Add noise
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Forward
            with torch.amp.autocast('cuda', enabled=args.mixed_precision and device.type == 'cuda'):
                noise_pred = model(noisy_latents, timesteps, dummy_context)
                loss = F.mse_loss(noise_pred, noise)

            # Backward
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Log
            losses.append(loss.item())
            global_step += 1
            pbar.update(1)

            if global_step % args.log_every == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            # Save checkpoint
            if global_step % args.save_every == 0:
                save_checkpoint(model, optimizer, global_step, args.output_dir)

    pbar.close()

    # Final save
    save_checkpoint(model, optimizer, global_step, args.output_dir, final=True)
    print(f"\nTraining complete! Final checkpoint saved to {args.output_dir}")


def save_checkpoint(model, optimizer, step, output_dir, final=False):
    """Save model checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if final:
        path = output_dir / "model_final.pt"
    else:
        path = output_dir / f"model_step_{step}.pt"

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

    # Also save config
    config_path = output_dir / "config.json"
    config = {
        'model_type': 'complexity-dit',
        'd_model': model.d_model,
        'num_layers': model.num_layers,
        'step': step,
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Complexity DiT')

    # Model
    parser.add_argument('--config', type=str, default='S', choices=['S', 'B', 'L', 'XL', 'XXL'],
                        help='Model config (S=100M, B=250M, L=500M, XL=700M)')
    parser.add_argument('--context_dim', type=int, default=768,
                        help='Context embedding dimension')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to image folder or latent folder')
    parser.add_argument('--use_latents', action='store_true',
                        help='Use pre-encoded latents')
    parser.add_argument('--vae_path', type=str, default=None,
                        help='Path to VAE weights (safetensors)')

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/dit')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10000)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
