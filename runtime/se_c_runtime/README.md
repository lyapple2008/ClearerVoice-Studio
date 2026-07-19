# MossFormer2_SE_48K C++ Runtime

该目录提供不依赖 Python 的 `MossFormer2_SE_48K` 推理程序：

- ONNX Runtime 执行神经网络；
- kaldi-native-fbank 生成与 torchaudio/Kaldi 对齐的 fbank；
- KissFFT 完成 1920 点 STFT/ISTFT；
- 固定 4 秒窗口、3 秒步长处理任意长度音频；
- `auto` 在 macOS 优先 CoreML，在 Windows/Linux 优先 CUDA，失败时回退 CPU。

## 当前输入范围

- 48 kHz RIFF/WAV；
- PCM 16/24/32 位或 float32；
- 支持多声道，各声道独立增强；
- 输出为 16 位 PCM WAV。

首版不包含重采样和 WAV 以外的编解码。非 48 kHz 音频应先转换为 48 kHz WAV。

## 1. 导出 ONNX

确保官方权重位于：

`clearvoice/checkpoints/MossFormer2_SE_48K/last_best_checkpoint.pt`

在已有 `clearer_voice` 环境安装导出依赖并执行：

```bash
conda run -n clearer_voice python -m pip install onnx onnxscript
conda run -n clearer_voice python runtime/se_c_runtime/tools/export_onnx.py
```

默认使用新版静态图导出器，生成约 232 MB 的单文件 ONNX。`--legacy` 仅用于排查兼容性；旧图不适合 CoreML。

## 2. 构建

需要 CMake 3.20+、C++17 编译器和 ONNX Runtime 开发包。CMake 会下载固定提交的 kaldi-native-fbank 及其 KissFFT 依赖。

### macOS：使用官方匹配发行包（推荐）

不要混用 Homebrew 头文件和 conda/Python wheel 动态库。以下命令下载本项目已验证的
ONNX Runtime 1.23.2 arm64 发行包：

```bash
curl -L --fail -o /tmp/onnxruntime-osx-arm64-1.23.2.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-arm64-1.23.2.tgz

echo "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856  /tmp/onnxruntime-osx-arm64-1.23.2.tgz" \
  | shasum -a 256 -c -

mkdir -p runtime/se_c_runtime/third_party
tar -xzf /tmp/onnxruntime-osx-arm64-1.23.2.tgz \
  -C runtime/se_c_runtime/third_party

cmake -S runtime/se_c_runtime -B runtime/se_c_runtime/build \
  -UONNXRUNTIME_INCLUDE_DIR \
  -UONNXRUNTIME_LIBRARY \
  -Uonnxruntime_DIR \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="$PWD/runtime/se_c_runtime/third_party/onnxruntime-osx-arm64-1.23.2"
cmake --build runtime/se_c_runtime/build -j
```

### 使用已安装的 CMake package

```bash
cmake -S runtime/se_c_runtime -B runtime/se_c_runtime/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/onnxruntime
cmake --build runtime/se_c_runtime/build -j
```

### 显式指定 ONNX Runtime

适用于官方预编译包、Python wheel 中的动态库，或没有 CMake config 的安装：

```bash
cmake -S runtime/se_c_runtime -B runtime/se_c_runtime/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DONNXRUNTIME_LIBRARY=/path/to/onnxruntime/library
cmake --build runtime/se_c_runtime/build -j
```

`ONNXRUNTIME_INCLUDE_DIR` 和 `ONNXRUNTIME_LIBRARY` 必须来自同一个 ONNX Runtime
发行包及版本。不能将 Homebrew 的头文件与 conda/Python wheel 的动态库混用；CMake
会在配置阶段执行 API 兼容性检查并拒绝这种组合。

macOS 使用 Homebrew 时必须确保 Homebrew 的 ONNX Runtime 及其全部依赖处于一致状态：

```bash
cmake -S runtime/se_c_runtime -B runtime/se_c_runtime/build \
  -UONNXRUNTIME_INCLUDE_DIR \
  -UONNXRUNTIME_LIBRARY \
  -UONNXRUNTIME_ROOT \
  -Uonnxruntime_DIR \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/onnxruntime
cmake --build runtime/se_c_runtime/build -j
```

ONNX Runtime 升级后必须重新运行上述 CMake 配置和构建命令，不能继续使用旧二进制。

Windows 使用 CUDA 版 ONNX Runtime 时，运行阶段还需确保 `onnxruntime.dll` 及 CUDA/cuDNN DLL 在 `PATH` 中。

## 3. 运行

```bash
runtime/se_c_runtime/build/clearvoice_se \
  runtime/se_c_runtime/models/mossformer2_se_48k.onnx \
  clearvoice/samples/test.wav \
  output.wav \
  --provider auto
```

`--provider` 可选值：`auto`、`cpu`、`coreml`、`cuda`。程序会对加速 provider 做一次非零/有限值自检；`auto` 自检失败时回退 CPU，显式指定 provider 时返回错误。

CoreML 第一次加载会编译 MLProgram，启动时间较长。部分 shape 运算保留在 CPU 属于 ONNX Runtime 的正常分区行为。

## 4. 数值验证

Python 基准关闭原实现中的随机 dither，并使用与 C++ 相同的固定窗口策略：

```bash
conda run -n clearer_voice python runtime/se_c_runtime/tools/reference.py \
  clearvoice/samples/test.wav /tmp/reference.wav

runtime/se_c_runtime/build/clearvoice_se \
  runtime/se_c_runtime/models/mossformer2_se_48k.onnx \
  clearvoice/samples/test.wav /tmp/candidate.wav \
  --provider cpu

conda run -n clearer_voice python runtime/se_c_runtime/tools/compare_audio.py \
  /tmp/reference.wav /tmp/candidate.wav
```

本机 `clearvoice/samples/test.wav` 验证结果：

| Provider | 最大 PCM 误差 | RMSE | SNR |
|---|---:|---:|---:|
| CPU | 1 LSB | 3.2281e-6 | 83.14 dB |
| CoreML | 1 LSB | 9.4180e-7 | 93.84 dB |

原 Python 代码调用 `fbank(..., dither=1.0)`，每次运行会加入随机噪声，因此无法直接要求逐样本完全相同。这里用 `dither=0` 建立可复现的数值基准。
