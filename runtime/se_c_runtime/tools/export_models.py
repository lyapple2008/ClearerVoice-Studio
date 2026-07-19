#!/usr/bin/env python3
"""Export ClearVoice models as fixed-window ONNX graphs for the C++ runtime."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "clearvoice_numba_cache")
)


MODEL_CONFIG = {
    "FRCRN_SE_16K": ("speech_enhancement", 16_000, 16_000),
    "MossFormerGAN_SE_16K": ("speech_enhancement", 16_000, 160_000),
    "MossFormer2_SS_16K": ("speech_separation", 16_000, 32_000),
    "MossFormer2_SR_48K": ("speech_super_resolution", 48_000, 192_000),
    "AV_MossFormer2_TSE_16K": ("target_speaker_extraction", 16_000, 48_000),
}


class FrcrnWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.model.inference_batch(audio)


class SeparationWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(audio)
        return outputs[0], outputs[1]


class GanReferenceWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, samples: int) -> None:
        super().__init__()
        self.model = model
        self.samples = samples
        self.register_buffer("window", torch.hamming_window(400, periodic=True))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(
            torch.tensor(float(self.samples), device=audio.device)
            / torch.clamp(torch.sum(audio**2, dim=-1, keepdim=True), min=1.0e-12)
        )
        normalized = audio * norm
        spectrum = torch.stft(
            normalized,
            n_fft=400,
            hop_length=100,
            win_length=400,
            window=self.window,
            center=True,
            onesided=True,
            return_complex=True,
        )
        real = spectrum.real
        imag = spectrum.imag
        magnitude = torch.clamp(real**2 + imag**2, min=1.0e-12).pow(0.15)
        unit = torch.rsqrt(torch.clamp(real**2 + imag**2, min=1.0e-12))
        compressed = torch.stack([magnitude * real * unit, magnitude * imag * unit], dim=1)
        compressed = compressed.permute(0, 1, 3, 2)
        pred_real, pred_imag = self.model(compressed)
        pred_real = pred_real.permute(0, 1, 3, 2).squeeze(1)
        pred_imag = pred_imag.permute(0, 1, 3, 2).squeeze(1)
        pred_power = torch.clamp(pred_real**2 + pred_imag**2, min=1.0e-12)
        pred_magnitude = pred_power.pow(1.0 / 0.6)
        pred_unit = torch.rsqrt(pred_power)
        complex_output = torch.complex(
            pred_magnitude * pred_real * pred_unit,
            pred_magnitude * pred_imag * pred_unit,
        )
        output = torch.istft(
            complex_output,
            n_fft=400,
            hop_length=100,
            win_length=400,
            window=self.window,
            center=True,
            onesided=True,
            length=self.samples,
        )
        return output / norm


class GanCoreWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, spectrum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(spectrum)


class SuperResolutionWrapper(torch.nn.Module):
    def __init__(self, models: torch.nn.ModuleList) -> None:
        super().__init__()
        import librosa

        self.mossformer = models[0]
        self.generator = models[1]
        mel = librosa.filters.mel(
            sr=48_000, n_fft=1024, n_mels=80, fmin=0, fmax=8000
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.register_buffer("window", torch.hann_window(1024))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(audio.unsqueeze(1), (384, 384), mode="reflect")
        spectrum = torch.stft(
            padded.squeeze(1),
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=self.window,
            center=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(spectrum.real**2 + spectrum.imag**2 + 1.0e-9)
        mel = torch.matmul(self.mel_basis, magnitude)
        mel = torch.log(torch.clamp(mel, min=1.0e-5))
        output = self.generator(self.mossformer(mel)).squeeze(1)
        return output


class AvWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        return self.model(audio, visual)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODEL_CONFIG))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-samples", type=int)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def load_model(repo_root: Path, task: str, name: str):
    sys.path.insert(0, str(repo_root / "clearvoice"))
    from clearvoice import ClearVoice

    previous = Path.cwd()
    try:
        os.chdir(repo_root / "clearvoice")
        clearvoice = ClearVoice(task=task, model_names=[name])
    finally:
        os.chdir(previous)
    model = clearvoice.models[0].model
    for module in model.modules():
        if hasattr(module, "cache_if_possible"):
            module.cache_if_possible = False
    return model.eval()


def build_wrapper(
    name: str, model, samples: int, *, reference: bool = False
) -> tuple[torch.nn.Module, tuple, list[str], list[str]]:
    audio = torch.zeros(1, samples, dtype=torch.float32)
    if name == "FRCRN_SE_16K":
        return FrcrnWrapper(model), (audio,), ["audio"], ["enhanced"]
    if name == "MossFormerGAN_SE_16K":
        if reference:
            return GanReferenceWrapper(model, samples), (audio,), ["audio"], ["enhanced"]
        frames = samples // 100 + 1
        spectrum = torch.zeros(1, 2, frames, 201, dtype=torch.float32)
        return (
            GanCoreWrapper(model),
            (spectrum,),
            ["spectrum"],
            ["pred_real", "pred_imag"],
        )
    if name == "MossFormer2_SS_16K":
        return SeparationWrapper(model), (audio,), ["audio"], ["speaker_1", "speaker_2"]
    if name == "MossFormer2_SR_48K":
        return SuperResolutionWrapper(model), (audio,), ["audio"], ["super_resolved"]
    if name == "AV_MossFormer2_TSE_16K":
        visual = torch.zeros(1, 75, 112, 112, dtype=torch.float32)
        return AvWrapper(model), (audio, visual), ["audio", "visual"], ["extracted"]
    raise ValueError(name)


def main() -> None:
    args = parse_args()
    task, _, default_samples = MODEL_CONFIG[args.model]
    samples = args.chunk_samples or default_samples
    output = args.output or (
        args.repo_root
        / "runtime/se_c_runtime/models"
        / f"{args.model.lower()}.onnx"
    )
    model = load_model(args.repo_root, task, args.model)
    wrapper, example, input_names, output_names = build_wrapper(args.model, model, samples)
    wrapper.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        dynamic_shapes = None
        use_dynamic = args.dynamic or args.model in {
            "FRCRN_SE_16K",
            "MossFormer2_SR_48K",
            "AV_MossFormer2_TSE_16K",
        }
        if use_dynamic:
            if args.model == "MossFormerGAN_SE_16K":
                dynamic_shapes = {
                    "spectrum": {2: torch.export.Dim("frames", min=5)}
                }
            elif args.model == "AV_MossFormer2_TSE_16K":
                dynamic_shapes = {
                    "audio": {1: torch.export.Dim("samples", min=16000)},
                    "visual": {
                        1: torch.export.Dim("visual_frames", min=25)
                    },
                }
            else:
                dynamic_shapes = {
                    "audio": {1: torch.export.Dim("samples", min=1024)}
                }
        torch.onnx.export(
            wrapper,
            example,
            output,
            input_names=input_names,
            output_names=output_names,
            opset_version=17 if args.legacy else 18,
            do_constant_folding=True,
            dynamo=not args.legacy,
            dynamic_shapes=dynamic_shapes,
            external_data=False,
        )
    print(f"exported {output}")


if __name__ == "__main__":
    main()
