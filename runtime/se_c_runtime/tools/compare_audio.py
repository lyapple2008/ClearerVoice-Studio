#!/usr/bin/env python3
"""Compare two PCM outputs and report numerical error."""

import argparse

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference, reference_rate = sf.read(args.reference, always_2d=True, dtype="float32")
    candidate, candidate_rate = sf.read(args.candidate, always_2d=True, dtype="float32")
    if reference_rate != candidate_rate or reference.shape != candidate.shape:
        raise ValueError("Sample rates and shapes must match")
    error = candidate - reference
    rmse = float(np.sqrt(np.mean(error**2)))
    signal_rms = float(np.sqrt(np.mean(reference**2)))
    snr = float(20.0 * np.log10(signal_rms / max(rmse, 1.0e-12)))
    print(f"max_abs={np.max(np.abs(error)):.8g}")
    print(f"mean_abs={np.mean(np.abs(error)):.8g}")
    print(f"rmse={rmse:.8g}")
    print(f"snr_db={snr:.4f}")


if __name__ == "__main__":
    main()
