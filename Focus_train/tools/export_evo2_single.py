from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nemo.export.utils.model_loader import load_model_weights


def export_checkpoint(source_dir: str, target_path: str) -> None:
    """Load a torch_dist checkpoint directory and flatten it into a single PT file."""
    state_dict = load_model_weights(source_dir)
    torch.save(state_dict, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten Evo-2 checkpoint to single weights.pt")
    parser.add_argument("--source", required=True, help="Path to checkpoints/evo2_7b_nemo directory")
    parser.add_argument("--target", required=True, help="Output file path (e.g. checkpoints/evo2_7b_nemo_converted/weights.pt)")
    args = parser.parse_args()

    source_dir = str(Path(args.source).expanduser())
    target_path = str(Path(args.target).expanduser())
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    export_checkpoint(source_dir, target_path)


if __name__ == "__main__":
    main()
