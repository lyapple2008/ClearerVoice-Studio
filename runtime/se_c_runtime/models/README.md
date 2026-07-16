# Generated model

Generate `mossformer2_se_48k.onnx` from the official checkpoint:

```bash
conda run -n clearer_voice python runtime/se_c_runtime/tools/export_onnx.py
```

The generated model uses a fixed input shape of `[1, 496, 180]`, corresponding
to one 4-second, 48 kHz audio chunk.
