#!/usr/bin/env python3
"""Generate deterministic PyTorch references for exported runtime models."""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from export_models import MODEL_CONFIG, build_wrapper, load_model


STRIDES = {
    "FRCRN_SE_16K": 12_000,
    "MossFormerGAN_SE_16K": 120_000,
    "MossFormer2_SS_16K": 24_000,
    "MossFormer2_SR_48K": 144_000,
    "AV_MossFormer2_TSE_16K": 28_800,
}


def source_path(path: Path, source: int) -> Path:
    return path.with_name(f"{path.stem}_s{source + 1}{path.suffix}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODEL_CONFIG))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--visual", type=Path)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def normalize_audio(audio: np.ndarray) -> tuple[np.ndarray, float]:
    eps = 1.0e-6
    target = 10 ** (-25 / 20)
    first = target / (np.sqrt(np.mean(audio**2)) + eps)
    normalized = audio * first
    power = normalized**2
    high_rms = np.sqrt(np.mean(power[power > np.mean(power)]))
    second = target / (high_rms + eps)
    return (normalized * second).astype(np.float32), 1 / (first * second + eps)


def python_padded_size(size: int, window: int, stride: int) -> int:
    if size < window:
        return window
    if size < window + stride:
        return window + stride
    if (size - window) % stride:
        return size + size - ((size - window) // stride) * stride
    return size


def main() -> None:
    args = parse_args()
    task, sample_rate, chunk_samples = MODEL_CONFIG[args.model]
    audio, rate = sf.read(args.input, dtype="float32")
    if rate != sample_rate or audio.ndim != 1:
        raise ValueError(f"Input must be mono {sample_rate} Hz WAV")

    model = load_model(args.repo_root, task, args.model)
    wrapper, _, _, output_names = build_wrapper(
        args.model, model, chunk_samples, reference=True
    )
    wrapper.eval()
    visual = None
    if args.visual:
        raw = np.fromfile(args.visual, dtype=np.float32)
        visual = raw.reshape(-1, 112, 112)

    stride = STRIDES[args.model]
    with torch.inference_mode():
        if args.model == "FRCRN_SE_16K":
            model_audio, restore = normalize_audio(audio)
            padded_size = python_padded_size(len(model_audio), chunk_samples, stride)
            tensor = torch.from_numpy(
                np.pad(model_audio, (0, padded_size - len(model_audio)))
            ).unsqueeze(0)
            outputs = [
                wrapper(tensor).detach().cpu().numpy().reshape(-1)[: len(audio)]
                * restore
            ]
        elif args.model == "MossFormer2_SR_48K":
            tensor = torch.from_numpy(audio).unsqueeze(0)
            outputs = [wrapper(tensor).detach().cpu().numpy().reshape(-1)]
        elif args.model == "AV_MossFormer2_TSE_16K" and len(audio) <= chunk_samples:
            if visual is None:
                raise ValueError("AV model requires --visual")
            peak = np.max(np.abs(audio))
            model_audio = audio / peak if peak > 0 else audio
            outputs = [
                wrapper(
                    torch.from_numpy(model_audio).unsqueeze(0),
                    torch.from_numpy(visual).unsqueeze(0),
                )
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            ]
        else:
            model_audio = audio
            if args.model == "AV_MossFormer2_TSE_16K":
                if visual is None:
                    raise ValueError("AV model requires --visual")
                peak = np.max(np.abs(model_audio))
                if peak > 0:
                    model_audio = model_audio / peak
            special_ss = args.model == "MossFormer2_SS_16K"
            if special_ss:
                model_audio, _ = normalize_audio(audio)
                padded_size = python_padded_size(
                    len(model_audio), chunk_samples, stride
                )
                chunks = 1 + (padded_size - chunk_samples) // stride
            else:
                chunks = (
                    1
                    if len(model_audio) <= chunk_samples
                    else 1
                    + (len(model_audio) - chunk_samples + stride - 1) // stride
                )
                padded_size = chunk_samples + (chunks - 1) * stride
            padded = np.pad(model_audio, (0, padded_size - len(model_audio)))
            outputs = [
                np.zeros(padded_size, dtype=np.float32) for _ in output_names
            ]
            give_up = (chunk_samples - stride) // 2
            for chunk in range(chunks):
                start = chunk * stride
                tensor = torch.from_numpy(
                    padded[start : start + chunk_samples]
                ).unsqueeze(0)
                if visual is None:
                    result = wrapper(tensor)
                else:
                    first_frame = round(start * 25 / sample_rate)
                    indices = np.minimum(
                        np.arange(first_frame, first_frame + 75), len(visual) - 1
                    )
                    visual_tensor = torch.from_numpy(visual[indices]).unsqueeze(0)
                    result = wrapper(tensor, visual_tensor)
                if not isinstance(result, tuple):
                    result = (result,)
                keep_begin = 0 if chunk == 0 else give_up
                keep_end = (
                    chunk_samples
                    if chunk + 1 == chunks
                    else chunk_samples - give_up
                )
                for index, value in enumerate(result):
                    samples = value.detach().cpu().numpy().reshape(-1)
                    copy_end = min(keep_end, len(samples))
                    outputs[index][start + keep_begin : start + copy_end] = (
                        samples[keep_begin:copy_end]
                    )
            if special_ss:
                input_rms = np.sqrt(np.mean(model_audio**2))
                for index, output in enumerate(outputs):
                    output_rms = np.sqrt(np.mean(output**2))
                    if output_rms > 1.0e-12:
                        outputs[index] = output / output_rms * input_rms
            outputs = [output[: len(model_audio)] for output in outputs]

    if args.model == "MossFormer2_SR_48K":
        sys.path.insert(0, str(args.repo_root / "clearvoice"))
        from clearvoice.utils.bandwidth_sub import bandwidth_sub

        outputs[0] = bandwidth_sub(audio, outputs[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, output in enumerate(outputs):
        path = args.output if len(outputs) == 1 else source_path(args.output, index)
        sf.write(path, output, sample_rate, subtype="PCM_16")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
