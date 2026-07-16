#!/usr/bin/env python3
"""Export the MossFormer2_SE_48K mask network to ONNX."""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch


class MaskModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)[0]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root
        / "clearvoice/checkpoints/MossFormer2_SE_48K/last_best_checkpoint.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "runtime/se_c_runtime/models/mossformer2_se_48k.onnx",
    )
    parser.add_argument("--frames", type=int, default=496)
    parser.add_argument("--legacy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "clearvoice/clearvoice"))

    from models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K

    model = MossFormer2_SE_48K(Namespace()).model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    if not args.legacy:
        for module in model.modules():
            if hasattr(module, "cache_if_possible"):
                module.cache_if_possible = False

    wrapper = MaskModel(model).eval()
    example = torch.zeros(1, args.frames, 180, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example,),
            args.output,
            input_names=["features"],
            output_names=["mask"],
            opset_version=17 if args.legacy else 18,
            do_constant_folding=True,
            dynamo=not args.legacy,
            external_data=False,
        )

    print(f"exported {args.output}")


if __name__ == "__main__":
    main()
