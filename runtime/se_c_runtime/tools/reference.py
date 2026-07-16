#!/usr/bin/env python3
"""Generate the deterministic PyTorch reference used by the C++ runtime."""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


SAMPLE_RATE = 48_000
CHUNK_SAMPLES = 192_000
STRIDE_SAMPLES = 144_000
GIVE_UP_SAMPLES = 24_000
MAX_WAV_VALUE = 32_768.0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root
        / "clearvoice/checkpoints/MossFormer2_SE_48K/last_best_checkpoint.pt",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "clearvoice/clearvoice"))
    from models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K

    model = MossFormer2_SE_48K(Namespace()).model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.eval()


def enhance_chunk(model: torch.nn.Module, normalized: np.ndarray) -> np.ndarray:
    audio = torch.from_numpy(normalized.astype(np.float32)) * MAX_WAV_VALUE
    fbanks = torchaudio.compliance.kaldi.fbank(
        audio.unsqueeze(0),
        dither=0.0,
        frame_length=40.0,
        frame_shift=8.0,
        num_mel_bins=60,
        sample_frequency=SAMPLE_RATE,
        window_type="hamming",
    )
    delta = torchaudio.functional.compute_deltas(fbanks.transpose(0, 1)).transpose(0, 1)
    delta_delta = torchaudio.functional.compute_deltas(delta.transpose(0, 1)).transpose(0, 1)
    features = torch.cat([fbanks, delta, delta_delta], dim=1).unsqueeze(0)
    mask = model(features)[0][0]

    window = torch.hamming_window(1920, periodic=False)
    spectrum = torch.stft(
        audio,
        n_fft=1920,
        hop_length=384,
        win_length=1920,
        window=window,
        center=False,
        return_complex=True,
    )
    enhanced = torch.istft(
        spectrum * mask.transpose(0, 1),
        n_fft=1920,
        hop_length=384,
        win_length=1920,
        window=window,
        center=False,
        length=CHUNK_SAMPLES,
    )
    return (enhanced / MAX_WAV_VALUE).numpy()


def enhance_channel(model: torch.nn.Module, audio: np.ndarray) -> np.ndarray:
    chunk_count = (
        1
        if len(audio) <= CHUNK_SAMPLES
        else 1 + (len(audio) - CHUNK_SAMPLES + STRIDE_SAMPLES - 1) // STRIDE_SAMPLES
    )
    padded_size = CHUNK_SAMPLES + (chunk_count - 1) * STRIDE_SAMPLES
    padded = np.pad(audio, (0, padded_size - len(audio)))
    output = np.zeros(padded_size, dtype=np.float32)
    for chunk in range(chunk_count):
        start = chunk * STRIDE_SAMPLES
        enhanced = enhance_chunk(model, padded[start : start + CHUNK_SAMPLES])
        keep_begin = 0 if chunk == 0 else GIVE_UP_SAMPLES
        keep_end = CHUNK_SAMPLES if chunk + 1 == chunk_count else CHUNK_SAMPLES - GIVE_UP_SAMPLES
        output[start + keep_begin : start + keep_end] = enhanced[keep_begin:keep_end]
    return output[: len(audio)]


def main() -> None:
    args = parse_args()
    audio, sample_rate = sf.read(args.input, always_2d=True, dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Input WAV must use a 48000 Hz sample rate")
    model = load_model(args.checkpoint)
    with torch.inference_mode():
        channels = [enhance_channel(model, audio[:, index]) for index in range(audio.shape[1])]
    output = np.stack(channels, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, output, SAMPLE_RATE, subtype="PCM_16")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
