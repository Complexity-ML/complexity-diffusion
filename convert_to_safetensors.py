"""
Convert PyTorch checkpoint to SafeTensors format.

Usage:
    python convert_to_safetensors.py model_step_20000.pt
    python convert_to_safetensors.py checkpoints/dit/model_step_20000.pt --output model.safetensors
"""

import argparse
from pathlib import Path
import torch
from safetensors.torch import save_file


def convert_to_safetensors(input_path: str, output_path: str = None):
    """Convert .pt checkpoint to .safetensors format."""
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix('.safetensors')
    else:
        output_path = Path(output_path)

    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        step = checkpoint.get('step', 'unknown')
        print(f"Checkpoint step: {step}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Convert to safetensors (requires contiguous tensors)
    safe_dict = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            safe_dict[key] = tensor.contiguous()

    print(f"Saving safetensors: {output_path}")
    print(f"  Tensors: {len(safe_dict)}")
    save_file(safe_dict, output_path)

    # Size comparison
    input_size = input_path.stat().st_size / 1e6
    output_size = output_path.stat().st_size / 1e6
    print(f"  Input size:  {input_size:.1f} MB")
    print(f"  Output size: {output_size:.1f} MB")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt to .safetensors")
    parser.add_argument("input", type=str, help="Input .pt file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output .safetensors file")
    args = parser.parse_args()

    convert_to_safetensors(args.input, args.output)
