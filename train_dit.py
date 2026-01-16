"""
Train Complexity DiT v0.3.0 - Mu-Guided Architecture for image generation.

v0.3.0 Features:
    - Mu-Guided KQV Attention (μ biases K, Q, V)
    - Mu-Guided Expert Routing (μ influences expert selection)
    - Contextual Mu (μ adapts based on hidden state)
    - Mu-Damped Dynamics (top-down guidance reduces oscillations)

Usage:
    # With HuggingFace dataset (recommended)
    python train_dit.py --dataset huggan/wikiart --batch_size 16 --steps 100000 --bf16

    # With local images
    python train_dit.py --data_dir /path/to/images --batch_size 16 --steps 100000

    # With pre-encoded latents (fastest)
    python train_dit.py --data_dir /path/to/latents --use_latents --batch_size 32
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


class HuggingFaceIterableDataset(torch.utils.data.IterableDataset):
    """HuggingFace streaming dataset - works with DataLoader num_workers."""

    def __init__(self, dataset_name: str, image_size: int = 256, split: str = "train"):
        from datasets import load_dataset
        from torchvision import transforms

        self.dataset_name = dataset_name
        self.split = split
        self.image_key = 'image'

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __iter__(self):
        from datasets import load_dataset
        from PIL import Image

        # Each worker gets its own iterator
        dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)

        for item in dataset:
            img = item[self.image_key]
            if not hasattr(img, 'convert'):
                img = Image.fromarray(img)
            img = img.convert('RGB')
            yield self.transform(img)


class HuggingFaceDataset(torch.utils.data.Dataset):
    """HuggingFace dataset wrapper (non-streaming)."""

    def __init__(self, dataset_name: str, image_size: int = 256, split: str = "train", max_samples: int = None):
        from datasets import load_dataset
        from torchvision import transforms

        print(f"Loading HuggingFace dataset: {dataset_name}...")

        self.dataset = load_dataset(dataset_name, split=split)
        if max_samples and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))
        print(f"Loaded {len(self.dataset)} samples")

        # Find image column
        self.image_key = None
        for key in ['image', 'img', 'images', 'pixel_values']:
            if key in self.dataset.column_names:
                self.image_key = key
                break
        if self.image_key is None:
            self.image_key = self.dataset.column_names[0]
        print(f"Using image column: {self.image_key}")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from PIL import Image
        img = self.dataset[idx][self.image_key]
        if not hasattr(img, 'convert'):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        return self.transform(img)


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
    if args.dataset:
        print(f"\nLoading HuggingFace dataset: {args.dataset}...")
        if args.streaming:
            # Streaming mode - use IterableDataset
            dataset = HuggingFaceIterableDataset(args.dataset, image_size=256)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        else:
            # Download mode
            dataset = HuggingFaceDataset(args.dataset, image_size=256, max_samples=args.max_samples)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
    elif args.use_latents:
        print(f"\nLoading latents from {args.data_dir}...")
        dataset = LatentDataset(args.data_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        print(f"\nLoading images from {args.data_dir}...")
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

    # Mixed precision (bf16 or fp16)
    if args.bf16 and device.type == 'cuda':
        # BF16 doesn't need GradScaler
        dtype = torch.bfloat16
        scaler = None
        print("Using BF16 precision")
    elif args.mixed_precision and device.type == 'cuda':
        dtype = torch.float16
        scaler = torch.amp.GradScaler('cuda')
        print("Using FP16 precision with GradScaler")
    else:
        dtype = torch.float32
        scaler = None
        print("Using FP32 precision")

    # Training loop
    print(f"\nStarting training for {args.steps} steps...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    model.train()
    global_step = 0
    losses = []

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

            # Create dummy context matching actual batch size
            actual_batch_size = latents.shape[0]
            dummy_context = torch.zeros(actual_batch_size, 77, args.context_dim, device=device)

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (actual_batch_size,), device=device)

            # Add noise
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Forward with autocast (bf16 or fp16)
            autocast_enabled = (args.bf16 or args.mixed_precision) and device.type == 'cuda'
            with torch.amp.autocast('cuda', enabled=autocast_enabled, dtype=dtype if autocast_enabled else None):
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
    parser.add_argument('--dataset', type=str, default=None,
                        help='HuggingFace dataset name (e.g., huggan/wikiart)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to image folder or latent folder')
    parser.add_argument('--use_latents', action='store_true',
                        help='Use pre-encoded latents')
    parser.add_argument('--vae_path', type=str, default=None,
                        help='Path to VAE weights (safetensors)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples to use from dataset')
    parser.add_argument('--streaming', action='store_true',
                        help='Stream dataset (no download)')

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use FP16 mixed precision')
    parser.add_argument('--bf16', action='store_true',
                        help='Use BF16 precision (recommended for H100)')
    parser.add_argument('--num_workers', type=int, default=4)

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/dit')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10000)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
