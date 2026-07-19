#include "gan_model.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <kiss_fftr.h>

#include "onnx_runner.h"

namespace {

constexpr int32_t kSampleRate = 16000;
constexpr size_t kChunkSamples = 160000;
constexpr size_t kStrideSamples = 120000;
constexpr size_t kGiveUpSamples = 20000;
constexpr int32_t kFftSize = 400;
constexpr int32_t kHopSize = 100;
constexpr int32_t kBins = 201;
constexpr int32_t kFrames = 1601;
constexpr int32_t kCenterPad = kFftSize / 2;
constexpr double kPi = 3.14159265358979323846;

std::vector<float> HammingWindow() {
  std::vector<float> window(kFftSize);
  for (int32_t i = 0; i < kFftSize; ++i) {
    window[i] = static_cast<float>(
        0.54 - 0.46 * std::cos(2.0 * kPi * i / kFftSize));
  }
  return window;
}

size_t ReflectedIndex(int64_t index, size_t size) {
  if (size < 2) return 0;
  while (index < 0 || index >= static_cast<int64_t>(size)) {
    if (index < 0) index = -index;
    if (index >= static_cast<int64_t>(size)) {
      index = 2 * static_cast<int64_t>(size) - index - 2;
    }
  }
  return static_cast<size_t>(index);
}

std::vector<float> CompressedSpectrum(const std::vector<float>& audio,
                                      float normalization) {
  const std::vector<float> window = HammingWindow();
  kiss_fftr_cfg config = kiss_fftr_alloc(kFftSize, 0, nullptr, nullptr);
  if (!config) throw std::runtime_error("Failed to allocate GAN FFT");
  std::vector<float> frame(kFftSize);
  std::vector<kiss_fft_cpx> bins(kBins);
  std::vector<float> output(2 * kFrames * kBins);

  for (int32_t frame_index = 0; frame_index < kFrames; ++frame_index) {
    const int64_t start = static_cast<int64_t>(frame_index * kHopSize) -
        kCenterPad;
    for (int32_t i = 0; i < kFftSize; ++i) {
      const size_t source = ReflectedIndex(start + i, audio.size());
      frame[i] = audio[source] * normalization * window[i];
    }
    kiss_fftr(config, frame.data(), bins.data());
    for (int32_t bin = 0; bin < kBins; ++bin) {
      const float real = bins[bin].r;
      const float imag = bins[bin].i;
      const float power = std::max(real * real + imag * imag, 1.0e-12f);
      const float magnitude = std::pow(power, 0.15f);
      const float inverse = 1.0f / std::sqrt(power);
      const size_t index = frame_index * kBins + bin;
      output[index] = magnitude * real * inverse;
      output[kFrames * kBins + index] = magnitude * imag * inverse;
    }
  }
  kiss_fft_free(config);
  return output;
}

std::vector<float> Reconstruct(const std::vector<float>& real,
                               const std::vector<float>& imag,
                               float normalization) {
  if (real.size() != static_cast<size_t>(kFrames * kBins) ||
      imag.size() != real.size()) {
    throw std::runtime_error("Unexpected MossFormerGAN ONNX output shape");
  }
  const std::vector<float> window = HammingWindow();
  const size_t padded_size = (kFrames - 1) * kHopSize + kFftSize;
  std::vector<float> padded(padded_size);
  std::vector<float> denominator(padded_size);
  std::vector<float> frame(kFftSize);
  std::vector<kiss_fft_cpx> bins(kBins);
  kiss_fftr_cfg config = kiss_fftr_alloc(kFftSize, 1, nullptr, nullptr);
  if (!config) throw std::runtime_error("Failed to allocate GAN inverse FFT");

  for (int32_t frame_index = 0; frame_index < kFrames; ++frame_index) {
    for (int32_t bin = 0; bin < kBins; ++bin) {
      const size_t index = frame_index * kBins + bin;
      const float power = std::max(
          real[index] * real[index] + imag[index] * imag[index], 1.0e-12f);
      const float magnitude = std::pow(power, 1.0f / 0.6f);
      const float inverse = 1.0f / std::sqrt(power);
      bins[bin].r = magnitude * real[index] * inverse;
      bins[bin].i = magnitude * imag[index] * inverse;
    }
    kiss_fftri(config, bins.data(), frame.data());
    const size_t start = static_cast<size_t>(frame_index * kHopSize);
    for (int32_t i = 0; i < kFftSize; ++i) {
      const float weighted = frame[i] * window[i] / kFftSize;
      padded[start + i] += weighted;
      denominator[start + i] += window[i] * window[i];
    }
  }
  kiss_fft_free(config);
  for (size_t i = 0; i < padded.size(); ++i) {
    if (denominator[i] > 1.0e-11f) padded[i] /= denominator[i];
  }
  std::vector<float> output(kChunkSamples);
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = padded[i + kCenterPad] / normalization;
  }
  return output;
}

float Normalization(const std::vector<float>& audio) {
  double energy = 0.0;
  for (float value : audio) energy += static_cast<double>(value) * value;
  return static_cast<float>(std::sqrt(
      static_cast<double>(audio.size()) / std::max(energy, 1.0e-12)));
}

}  // namespace

class MossFormerGanModel::Impl {
 public:
  Impl(const std::string& model_path, const std::string& provider)
      : runner(model_path, provider) {
    if (runner.InputCount() != 1 || runner.OutputCount() != 2) {
      throw std::runtime_error(
          "MossFormerGAN ONNX model must have one input and two outputs");
    }
  }

  std::vector<float> ProcessChunk(const std::vector<float>& audio) const {
    const float normalization = Normalization(audio);
    TensorInput input;
    input.data = CompressedSpectrum(audio, normalization);
    input.shape = {1, 2, kFrames, kBins};
    const auto outputs = runner.Run({input});
    return Reconstruct(outputs[0].data, outputs[1].data, normalization);
  }

  std::vector<float> Process(const std::vector<float>& audio) const {
    if (audio.empty()) return {};
    const size_t chunks = audio.size() <= kChunkSamples
        ? 1
        : 1 + (audio.size() - kChunkSamples + kStrideSamples - 1) /
                  kStrideSamples;
    const size_t padded_size = kChunkSamples + (chunks - 1) * kStrideSamples;
    std::vector<float> padded = audio;
    padded.resize(padded_size);
    std::vector<float> output(padded_size);
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
      const size_t start = chunk * kStrideSamples;
      std::vector<float> input(padded.begin() + start,
                               padded.begin() + start + kChunkSamples);
      const auto enhanced = ProcessChunk(input);
      const size_t keep_begin = chunk == 0 ? 0 : kGiveUpSamples;
      const size_t keep_end = chunk + 1 == chunks
          ? kChunkSamples
          : kChunkSamples - kGiveUpSamples;
      std::copy(enhanced.begin() + keep_begin, enhanced.begin() + keep_end,
                output.begin() + start + keep_begin);
    }
    output.resize(audio.size());
    return output;
  }

  OnnxRunner runner;
};

MossFormerGanModel::MossFormerGanModel(const std::string& model_path,
                                       const std::string& provider)
    : impl_(std::make_unique<Impl>(model_path, provider)) {}

MossFormerGanModel::~MossFormerGanModel() = default;

int32_t MossFormerGanModel::SampleRate() const { return kSampleRate; }

size_t MossFormerGanModel::OutputCount() const { return 1; }

std::vector<std::vector<float>> MossFormerGanModel::Process(
    const ClearVoiceInput& input) const {
  return {impl_->Process(input.audio)};
}

const std::string& MossFormerGanModel::ActiveProvider() const {
  return impl_->runner.ActiveProvider();
}
