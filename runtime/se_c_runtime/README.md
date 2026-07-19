# ClearVoice C++ Runtime

该目录提供不依赖 Python 的 ClearVoice 推理程序。C++ API 通过
`ClearVoice::Create(ModelType, model_path, provider)` 创建对应模型子类，统一处理
分块、重叠裁剪、多说话人输出、采样率转换和 ONNX Runtime provider。

支持 `clearvoice/demo.py` 中的全部模型类型：

| ModelType | ONNX 输入 | C++ 前后处理 |
|---|---|---|
| `FRCRN_SE_16K` | 动态长度波形 | Python 对齐的幅度归一化、补齐与反归一化 |
| `MossFormer2_SE_48K` | 496x180 fbank | fbank、delta、STFT/ISTFT、4 秒滑窗 |
| `MossFormerGAN_SE_16K` | 压缩复数频谱 | 归一化、STFT、功率压缩/解压、ISTFT |
| `MossFormer2_SS_16K` | 2 秒波形 | Python 对齐的特殊补齐、滑窗、双说话人 RMS |
| `MossFormer2_SR_48K` | 动态长度波形 | 重采样、`bandwidth_sub` Butterworth 后处理 |
| `AV_MossFormer2_TSE_16K` | 动态波形 + 灰度人脸帧 | 短音频原长推理、长音频 3 秒滑窗 |

底层实现包括：

- ONNX Runtime 执行神经网络；
- kaldi-native-fbank 生成与 torchaudio/Kaldi 对齐的 fbank；
- KissFFT 完成模型所需的 STFT/ISTFT 和带宽检测；
- `auto` 在 macOS 优先 CoreML，在 Windows/Linux 优先 CUDA，失败时回退 CPU。

## 当前输入范围

- RIFF/WAV；
- PCM 16/24/32 位或 float32；
- 支持多声道，各声道独立增强；
- 输入采样率不匹配时由 C++ windowed-sinc 重采样；
- 输出为 16 位 PCM WAV。

WAV 以外的编解码不属于 pure C++ runtime；可在调用前用 FFmpeg 转为 WAV。
`MossFormer2_SR_48K` 输出固定为 48 kHz，其余模型恢复输入文件采样率。

## 1. 导出 ONNX

确保官方权重位于：

`clearvoice/checkpoints/MossFormer2_SE_48K/last_best_checkpoint.pt`

在已有 `clearer_voice` 环境安装导出依赖并执行：

```bash
conda run -n clearer_voice python -m pip install onnx onnxscript
conda run -n clearer_voice python runtime/se_c_runtime/tools/export_onnx.py
```

默认使用新版静态图导出器，生成约 232 MB 的单文件 ONNX。`--legacy` 仅用于排查兼容性；旧图不适合 CoreML。

其余模型使用统一导出器：

```bash
for model in \
  FRCRN_SE_16K \
  MossFormerGAN_SE_16K \
  MossFormer2_SS_16K \
  MossFormer2_SR_48K \
  AV_MossFormer2_TSE_16K
do
  conda run -n clearer_voice python \
    runtime/se_c_runtime/tools/export_models.py "$model"
done
```

导出器复用 Python ClearVoice 的官方权重加载逻辑，缺少 checkpoint 时会从官方
Hugging Face 仓库下载。FRCRN、SR 和 AV 默认导出动态长度图；其余模型使用与 Python
滑窗一致的固定图。模型文件位于 `runtime/se_c_runtime/models/`，不提交到 Git。

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

统一 CLI：

```bash
runtime/se_c_runtime/build/clearvoice \
  MODEL_TYPE MODEL.onnx INPUT.wav OUTPUT.wav \
  --provider cpu
```

分离模型自动写为 `OUTPUT_s1.wav` 和 `OUTPUT_s2.wav`。例如：

```bash
runtime/se_c_runtime/build/clearvoice \
  MossFormer2_SS_16K \
  runtime/se_c_runtime/models/mossformer2_ss_16k.onnx \
  clearvoice/samples/input_ss.wav \
  /tmp/separated.wav \
  --provider cpu
```

AV 模型额外接收预处理视觉张量：

```bash
runtime/se_c_runtime/build/clearvoice \
  AV_MossFormer2_TSE_16K \
  runtime/se_c_runtime/models/av_mossformer2_tse_16k.onnx \
  input_audio.wav output.wav \
  --visual face_frames.f32 \
  --provider cpu
```

`face_frames.f32` 是连续 float32 灰度帧，单帧 112x112，25 fps，归一化公式与
Python 相同：`(gray / 255 - 0.4161) / 0.1688`。人脸检测、跟踪和裁剪属于视频
前处理，不依赖神经网络 runtime；可复用 `clearvoice/utils/video_process.py` 产生的
face track。

原有单模型命令保持兼容：

```bash
runtime/se_c_runtime/build/clearvoice_se \
  runtime/se_c_runtime/models/mossformer2_se_48k.onnx \
  clearvoice/samples/test.wav \
  output.wav \
  --provider auto
```

`--provider` 可选值：`auto`、`cpu`、`coreml`、`cuda`。`auto` 在加速 provider
无法创建会话时回退 CPU；显式指定 provider 时直接返回错误。SE 专用 runtime 还会
执行一次非零/有限值自检。

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

其他模型使用 `tools/reference_models.py` 生成与 C++ 分块策略一致的 PyTorch 基准。例如：

```bash
conda run -n clearer_voice python \
  runtime/se_c_runtime/tools/reference_models.py \
  MossFormer2_SS_16K clearvoice/samples/input_ss.wav /tmp/reference.wav

runtime/se_c_runtime/build/clearvoice \
  MossFormer2_SS_16K \
  runtime/se_c_runtime/models/mossformer2_ss_16k.onnx \
  clearvoice/samples/input_ss.wav /tmp/candidate.wav \
  --provider cpu
```

本机 CPU 对齐结果：

| 模型 | 基准 | SNR |
|---|---|---:|
| `MossFormer2_SE_48K` | deterministic PyTorch | 83.14 dB |
| `FRCRN_SE_16K` | Python demo | 68.88 dB |
| `MossFormerGAN_SE_16K` | fixed-window PyTorch | 76.60 dB |
| `MossFormer2_SS_16K` speaker 1 / 2 | Python demo | 64.57 / 64.01 dB |
| `MossFormer2_SR_48K` | same-rate Python final output | 90.67 dB |
| `AV_MossFormer2_TSE_16K` | PyTorch, dynamic synthetic visual tensor | 76.06 dB |

SR 的 16 kHz demo 输入会经过 C++ 与 librosa/soxr 不同的重采样器；波形不逐样本
等同，但模型输出采样率、长度和处理流程一致。要求逐样本比较时，应先将输入统一
转换为 48 kHz WAV。
