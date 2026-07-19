# Generated models

Generate `mossformer2_se_48k.onnx` from the official checkpoint:

```bash
conda run -n clearer_voice python runtime/se_c_runtime/tools/export_onnx.py
```

The generated model uses a fixed input shape of `[1, 496, 180]`, corresponding
to one 4-second, 48 kHz audio chunk.

Export the remaining ClearVoice models with:

```bash
conda run -n clearer_voice python runtime/se_c_runtime/tools/export_models.py MODEL_TYPE
```

Supported `MODEL_TYPE` values are `FRCRN_SE_16K`,
`MossFormerGAN_SE_16K`, `MossFormer2_SS_16K`, `MossFormer2_SR_48K`, and
`AV_MossFormer2_TSE_16K`. The exporter downloads the official checkpoint through
the existing Python ClearVoice loader when it is not already present.

FRCRN, SR and AV use dynamic input lengths by default. MossFormerGAN exports only
its neural spectrum core; STFT, normalization, power compression and ISTFT are
implemented in C++. SS exports a fixed window matching its Python segment size.
