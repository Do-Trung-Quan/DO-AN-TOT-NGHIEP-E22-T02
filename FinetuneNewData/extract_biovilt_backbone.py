"""Extract the fine-tuned BioViL-T backbone from a classifier checkpoint."""

import argparse
import os

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(HERE, "best_model_biovilt_silicosis_ce.pth")
DEFAULT_OUTPUT = os.path.join(HERE, "biovilt_silicosis_ce_backbone.pth")


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Extract backbone weights from a BioViL-T classifier checkpoint."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a PyTorch state_dict.")

    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in checkpoint.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError(
            "No keys starting with 'backbone.' were found. "
            "This may already be a backbone-only checkpoint."
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(backbone_state, args.output)
    print(f"Extracted {len(backbone_state)} backbone tensors")
    print(f"Input : {args.checkpoint}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
