#include "enhancer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <kiss_fftr.h>
#include <onnxruntime_cxx_api.h>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

namespace {

constexpr int32_t kSampleRate = 48000;
constexpr int32_t kChunkSamples = 192000;
constexpr int32_t kStrideSamples = 144000;
constexpr int32_t kGiveUpSamples = 24000;
constexpr int32_t kFftSize = 1920;
constexpr int32_t kHopSize = 384;
constexpr int32_t kBins = kFftSize / 2 + 1;
constexpr int32_t kFrames = 496;
constexpr int32_t kMelBins = 60;
constexpr int32_t kFeatureDim = 180;
constexpr float kMaxWavValue = 32768.0f;
constexpr float kPi = 3.14159265358979323846f;

std::vector<float> HammingWindow() {
  std::vector<float> window(kFftSize);
  for (int32_t i = 0; i < kFftSize; ++i) {
    window[i] = 0.54f - 0.46f *
        std::cos(2.0f * kPi * i / static_cast<float>(kFftSize - 1));
  }
  return window;
}

std::vector<float> ComputeFeatures(const std::vector<float>& audio) {
  knf::FbankOptions options;
  options.frame_opts.samp_freq = kSampleRate;
  options.frame_opts.frame_length_ms = 40.0f;
  options.frame_opts.frame_shift_ms = 8.0f;
  options.frame_opts.dither = 0.0f;
  options.frame_opts.preemph_coeff = 0.97f;
  options.frame_opts.remove_dc_offset = true;
  options.frame_opts.window_type = "hamming";
  options.frame_opts.round_to_power_of_two = true;
  options.frame_opts.snip_edges = true;
  options.mel_opts.num_bins = kMelBins;
  options.mel_opts.low_freq = 20.0f;
  options.mel_opts.high_freq = 0.0f;
  options.use_energy = false;
  options.use_log_fbank = true;
  options.use_power = true;

  knf::OnlineFbank fbank(options);
  fbank.AcceptWaveform(kSampleRate, audio.data(), audio.size());
  fbank.InputFinished();
  if (fbank.NumFramesReady() != kFrames) {
    throw std::runtime_error("Unexpected fbank frame count");
  }

  std::vector<float> base(kFrames * kMelBins);
  for (int32_t frame = 0; frame < kFrames; ++frame) {
    const float* values = fbank.GetFrame(frame);
    std::copy(values, values + kMelBins, base.begin() + frame * kMelBins);
  }

  auto delta = [](const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (int32_t frame = 0; frame < kFrames; ++frame) {
      for (int32_t bin = 0; bin < kMelBins; ++bin) {
        float value = 0.0f;
        for (int32_t offset = 1; offset <= 2; ++offset) {
          const int32_t left = std::max(0, frame - offset);
          const int32_t right = std::min(kFrames - 1, frame + offset);
          value += offset * (input[right * kMelBins + bin] -
                             input[left * kMelBins + bin]);
        }
        output[frame * kMelBins + bin] = value / 10.0f;
      }
    }
    return output;
  };

  const std::vector<float> first = delta(base);
  const std::vector<float> second = delta(first);
  std::vector<float> features(kFrames * kFeatureDim);
  for (int32_t frame = 0; frame < kFrames; ++frame) {
    std::copy_n(base.data() + frame * kMelBins, kMelBins,
                features.data() + frame * kFeatureDim);
    std::copy_n(first.data() + frame * kMelBins, kMelBins,
                features.data() + frame * kFeatureDim + kMelBins);
    std::copy_n(second.data() + frame * kMelBins, kMelBins,
                features.data() + frame * kFeatureDim + 2 * kMelBins);
  }
  return features;
}

std::vector<std::complex<float>> Stft(const std::vector<float>& audio) {
  const std::vector<float> window = HammingWindow();
  kiss_fftr_cfg config = kiss_fftr_alloc(kFftSize, 0, nullptr, nullptr);
  if (!config) throw std::runtime_error("Failed to allocate FFT");
  std::vector<float> frame(kFftSize);
  std::vector<kiss_fft_cpx> spectrum(kBins);
  std::vector<std::complex<float>> output(kFrames * kBins);
  for (int32_t index = 0; index < kFrames; ++index) {
    const int32_t start = index * kHopSize;
    for (int32_t i = 0; i < kFftSize; ++i) {
      frame[i] = audio[start + i] * window[i];
    }
    kiss_fftr(config, frame.data(), spectrum.data());
    for (int32_t bin = 0; bin < kBins; ++bin) {
      output[index * kBins + bin] =
          {spectrum[bin].r, spectrum[bin].i};
    }
  }
  kiss_fft_free(config);
  return output;
}

std::vector<float> Istft(const std::vector<std::complex<float>>& spectrum) {
  const std::vector<float> window = HammingWindow();
  kiss_fftr_cfg config = kiss_fftr_alloc(kFftSize, 1, nullptr, nullptr);
  if (!config) throw std::runtime_error("Failed to allocate inverse FFT");
  std::vector<float> output(kChunkSamples);
  std::vector<float> denominator(kChunkSamples);
  std::vector<float> frame(kFftSize);
  std::vector<kiss_fft_cpx> bins(kBins);

  for (int32_t index = 0; index < kFrames; ++index) {
    for (int32_t bin = 0; bin < kBins; ++bin) {
      const auto value = spectrum[index * kBins + bin];
      bins[bin].r = value.real();
      bins[bin].i = value.imag();
    }
    kiss_fftri(config, bins.data(), frame.data());
    const int32_t start = index * kHopSize;
    for (int32_t i = 0; i < kFftSize; ++i) {
      const float weighted = frame[i] * window[i] / kFftSize;
      output[start + i] += weighted;
      denominator[start + i] += window[i] * window[i];
    }
  }
  kiss_fft_free(config);
  for (size_t i = 0; i < output.size(); ++i) {
    if (denominator[i] > 1.0e-11f) output[i] /= denominator[i];
  }
  return output;
}

}  // namespace

class Enhancer::Impl {
 public:
  Impl(const std::string& model_path, const std::string& requested_provider) {
    const OrtApiBase* api_base = OrtGetApiBase();
    if (api_base == nullptr || api_base->GetApi(ORT_API_VERSION) == nullptr) {
      const std::string runtime_version =
          api_base == nullptr ? "unknown" : api_base->GetVersionString();
      throw std::runtime_error(
          "ONNX Runtime API mismatch: headers request API " +
          std::to_string(ORT_API_VERSION) + ", runtime version is " +
          runtime_version + ". Rebuild with matching headers and library.");
    }
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                     "clearvoice_se");
    const std::filesystem::path ort_model_path(model_path);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    ConfigureProvider(requested_provider);
    try {
      session = std::make_unique<Ort::Session>(
          *env, ort_model_path.c_str(), options);
      if (active_provider != "cpu" && !ProviderSelfTest()) {
        throw std::runtime_error(active_provider +
                                 " provider returned an invalid all-zero mask");
      }
    } catch (const std::exception&) {
      if (requested_provider != "auto") throw;
      session.reset();
      options = Ort::SessionOptions{};
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      active_provider = "cpu";
      session = std::make_unique<Ort::Session>(
          *env, ort_model_path.c_str(), options);
    }
  }

  bool ProviderSelfTest() const {
    std::vector<float> features(kFrames * kFeatureDim, 0.0f);
    const std::array<int64_t, 3> shape = {1, kFrames, kFeatureDim};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory, features.data(), features.size(), shape.data(), shape.size());
    const char* input_names[] = {"features"};
    const char* output_names[] = {"mask"};
    auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                output_names, 1);
    const float* mask = outputs.front().GetTensorData<float>();
    const size_t count = kFrames * kBins;
    for (size_t i = 0; i < count; ++i) {
      if (!std::isfinite(mask[i])) return false;
      if (std::abs(mask[i]) > 1.0e-6f) return true;
    }
    return false;
  }

  void ConfigureProvider(const std::string& requested) {
    if (requested != "auto" && requested != "cpu" && requested != "coreml" &&
        requested != "cuda") {
      throw std::runtime_error("Provider must be auto, cpu, coreml, or cuda");
    }
    if (requested == "cpu") {
      active_provider = "cpu";
      return;
    }

#ifdef __APPLE__
    if (requested == "auto" || requested == "coreml") {
      try {
        options.AppendExecutionProvider(
            "CoreML", {{"ModelFormat", "MLProgram"},
                       {"RequireStaticInputShapes", "1"}});
        active_provider = "coreml";
        return;
      } catch (const Ort::Exception&) {
        if (requested == "coreml") throw;
      }
    }
#endif

    if (requested == "auto" || requested == "cuda") {
      try {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda_options);
        active_provider = "cuda";
        return;
      } catch (const Ort::Exception&) {
        if (requested == "cuda") throw;
      }
    }
    active_provider = "cpu";
  }

  std::vector<float> ProcessChunk(const std::vector<float>& normalized) const {
    std::vector<float> scaled(kChunkSamples);
    std::transform(normalized.begin(), normalized.end(), scaled.begin(),
                   [](float value) { return value * kMaxWavValue; });
    std::vector<float> features = ComputeFeatures(scaled);

    const std::array<int64_t, 3> input_shape = {1, kFrames, kFeatureDim};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory, features.data(), features.size(), input_shape.data(),
        input_shape.size());
    const char* input_names[] = {"features"};
    const char* output_names[] = {"mask"};
    auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                output_names, 1);
    const float* mask = outputs.front().GetTensorData<float>();

    std::vector<std::complex<float>> spectrum = Stft(scaled);
    for (int32_t frame = 0; frame < kFrames; ++frame) {
      for (int32_t bin = 0; bin < kBins; ++bin) {
        spectrum[frame * kBins + bin] *= mask[frame * kBins + bin];
      }
    }
    std::vector<float> output = Istft(spectrum);
    std::transform(output.begin(), output.end(), output.begin(),
                   [](float value) { return value / kMaxWavValue; });
    return output;
  }

  std::vector<float> Process(const std::vector<float>& audio) const {
    if (audio.empty()) return {};
    const size_t chunk_count = audio.size() <= kChunkSamples
        ? 1
        : 1 + (audio.size() - kChunkSamples + kStrideSamples - 1) /
                  kStrideSamples;
    const size_t padded_size = kChunkSamples + (chunk_count - 1) * kStrideSamples;
    std::vector<float> padded = audio;
    padded.resize(padded_size);
    std::vector<float> output(padded_size);

    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
      const size_t start = chunk * kStrideSamples;
      std::vector<float> input(padded.begin() + start,
                               padded.begin() + start + kChunkSamples);
      const std::vector<float> enhanced = ProcessChunk(input);
      const size_t keep_begin = chunk == 0 ? 0 : kGiveUpSamples;
      const size_t keep_end = chunk + 1 == chunk_count
          ? kChunkSamples
          : kChunkSamples - kGiveUpSamples;
      std::copy(enhanced.begin() + keep_begin, enhanced.begin() + keep_end,
                output.begin() + start + keep_begin);
    }
    output.resize(audio.size());
    return output;
  }

  std::unique_ptr<Ort::Env> env;
  Ort::SessionOptions options;
  std::unique_ptr<Ort::Session> session;
  std::string active_provider = "cpu";
};

Enhancer::Enhancer(const std::string& model_path, const std::string& provider)
    : impl_(std::make_unique<Impl>(model_path, provider)) {}

Enhancer::~Enhancer() = default;

std::vector<float> Enhancer::Process(const std::vector<float>& audio) const {
  return impl_->Process(audio);
}

const std::string& Enhancer::ActiveProvider() const {
  return impl_->active_provider;
}
