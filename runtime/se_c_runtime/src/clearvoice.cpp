#include "clearvoice.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "bandwidth_sub.h"
#include "enhancer.h"
#include "gan_model.h"
#include "onnx_runner.h"

namespace {

struct ModelConfig {
  int32_t sample_rate;
  size_t chunk_samples;
  size_t stride_samples;
  size_t output_count;
  bool normalize_peak;
  bool normalize_input;
  bool restore_input_scale;
  bool normalize_outputs;
  bool substitute_bandwidth;
  size_t direct_limit_samples;
  bool pad_direct_input;
  bool crop_direct_output;
  bool python_segment_padding;
  bool normalize_before_crop;
  size_t visual_frames;
  size_t visual_height;
  size_t visual_width;
};

float Rms(const std::vector<float>& audio) {
  if (audio.empty()) return 0.0f;
  double energy = 0.0;
  for (float value : audio) energy += static_cast<double>(value) * value;
  return static_cast<float>(std::sqrt(energy / audio.size()));
}

std::pair<std::vector<float>, float> NormalizeInput(
    const std::vector<float>& audio) {
  constexpr float kEpsilon = 1.0e-6f;
  constexpr float kTarget = 0.0562341325f;
  std::vector<float> normalized = audio;
  const float first = kTarget / (Rms(normalized) + kEpsilon);
  for (float& value : normalized) value *= first;
  double high_energy = 0.0;
  size_t high_count = 0;
  double average_power = 0.0;
  for (float value : normalized) average_power += value * value;
  average_power /= normalized.size();
  for (float value : normalized) {
    const double power = value * value;
    if (power > average_power) {
      high_energy += power;
      ++high_count;
    }
  }
  const float high_rms = high_count == 0
      ? 0.0f
      : static_cast<float>(std::sqrt(high_energy / high_count));
  const float second = kTarget / (high_rms + kEpsilon);
  for (float& value : normalized) value *= second;
  return {std::move(normalized), 1.0f / (first * second + kEpsilon)};
}

class MossFormer2SeModel final : public ClearVoice {
 public:
  MossFormer2SeModel(const std::string& model_path,
                     const std::string& provider)
      : enhancer_(model_path, provider) {}

  int32_t SampleRate() const override { return 48000; }
  size_t OutputCount() const override { return 1; }
  std::vector<std::vector<float>> Process(
      const ClearVoiceInput& input) const override {
    return {enhancer_.Process(input.audio)};
  }
  const std::string& ActiveProvider() const override {
    return enhancer_.ActiveProvider();
  }

 private:
  Enhancer enhancer_;
};

class DirectOnnxModel final : public ClearVoice {
 public:
  DirectOnnxModel(ModelConfig config, const std::string& model_path,
                  const std::string& provider)
      : config_(config), runner_(model_path, provider) {
    const size_t expected_inputs = config_.visual_frames == 0 ? 1 : 2;
    if (runner_.InputCount() != expected_inputs) {
      throw std::runtime_error("Unexpected ONNX input count for selected model");
    }
  }

  int32_t SampleRate() const override { return config_.sample_rate; }
  size_t OutputCount() const override { return config_.output_count; }
  const std::string& ActiveProvider() const override {
    return runner_.ActiveProvider();
  }

  std::vector<std::vector<float>> Process(
      const ClearVoiceInput& input) const override {
    if (input.audio.empty()) return std::vector<std::vector<float>>(
        config_.output_count);
    ValidateVisual(input);
    std::vector<float> model_audio = input.audio;
    float restore_scale = 1.0f;
    if (config_.normalize_peak) {
      float peak = 0.0f;
      for (float value : model_audio) peak = std::max(peak, std::abs(value));
      if (peak > 0.0f) {
        for (float& value : model_audio) value /= peak;
      }
    }
    if (config_.normalize_input) {
      auto normalized = NormalizeInput(input.audio);
      model_audio = std::move(normalized.first);
      if (config_.restore_input_scale) restore_scale = normalized.second;
    }
    if (config_.direct_limit_samples != 0 &&
        model_audio.size() <= config_.direct_limit_samples) {
      TensorInput audio_input;
      audio_input.data = model_audio;
      if (config_.pad_direct_input) {
        const size_t size = audio_input.data.size();
        size_t padded_size = size;
        if (size < config_.chunk_samples) {
          padded_size = config_.chunk_samples;
        } else if (size < config_.chunk_samples + config_.stride_samples) {
          padded_size = config_.chunk_samples + config_.stride_samples;
        } else if ((size - config_.chunk_samples) % config_.stride_samples != 0) {
          const size_t padding = size -
              ((size - config_.chunk_samples) / config_.stride_samples) *
                  config_.stride_samples;
          padded_size += padding;
        }
        audio_input.data.resize(padded_size);
      }
      audio_input.shape = {1, static_cast<int64_t>(audio_input.data.size())};
      std::vector<TensorInput> direct_inputs;
      direct_inputs.push_back(std::move(audio_input));
      if (config_.visual_frames != 0) {
        TensorInput visual_input;
        visual_input.data = input.visual;
        visual_input.shape = {1, input.visual_frames, input.visual_height,
                              input.visual_width};
        direct_inputs.push_back(std::move(visual_input));
      }
      auto result = SplitOutputs(runner_.Run(direct_inputs));
      if (config_.crop_direct_output) {
        for (auto& output : result) {
          output.resize(std::min(output.size(), model_audio.size()));
        }
      }
      if (config_.substitute_bandwidth) {
        result[0] = SubstituteBandwidth(input.audio, result[0],
                                        config_.sample_rate);
      }
      if (restore_scale != 1.0f) {
        for (auto& output : result) {
          for (float& value : output) value *= restore_scale;
        }
      }
      return result;
    }

    size_t padded_size = model_audio.size();
    size_t chunks = 1;
    if (config_.python_segment_padding) {
      if (padded_size < config_.chunk_samples) {
        padded_size = config_.chunk_samples;
      } else if (padded_size <
                 config_.chunk_samples + config_.stride_samples) {
        padded_size = config_.chunk_samples + config_.stride_samples;
      } else if ((padded_size - config_.chunk_samples) %
                     config_.stride_samples != 0) {
        padded_size += padded_size -
            ((padded_size - config_.chunk_samples) / config_.stride_samples) *
                config_.stride_samples;
      }
      chunks = 1 + (padded_size - config_.chunk_samples) /
          config_.stride_samples;
    } else {
      chunks = model_audio.size() <= config_.chunk_samples
          ? 1
          : 1 + (model_audio.size() - config_.chunk_samples +
                 config_.stride_samples - 1) / config_.stride_samples;
      padded_size = config_.chunk_samples +
          (chunks - 1) * config_.stride_samples;
    }
    std::vector<float> padded = model_audio;
    padded.resize(padded_size);
    std::vector<std::vector<float>> result(
        config_.output_count, std::vector<float>(padded_size));
    const size_t give_up =
        (config_.chunk_samples - config_.stride_samples) / 2;

    for (size_t chunk = 0; chunk < chunks; ++chunk) {
      const size_t start = chunk * config_.stride_samples;
      TensorInput audio_input;
      audio_input.data.assign(padded.begin() + start,
                              padded.begin() + start + config_.chunk_samples);
      audio_input.shape = {1, static_cast<int64_t>(config_.chunk_samples)};
      std::vector<TensorInput> inputs;
      inputs.push_back(std::move(audio_input));
      if (config_.visual_frames != 0) {
        inputs.push_back(MakeVisualChunk(input, start));
      }
      const auto outputs = SplitOutputs(runner_.Run(inputs));
      const size_t keep_begin = chunk == 0 ? 0 : give_up;
      const size_t keep_end = chunk + 1 == chunks
          ? config_.chunk_samples
          : config_.chunk_samples - give_up;
      for (size_t output = 0; output < config_.output_count; ++output) {
        const size_t available = std::min(config_.chunk_samples,
                                           outputs[output].size());
        const size_t copy_end = std::min(keep_end, available);
        if (copy_end > keep_begin) {
          std::copy(outputs[output].begin() + keep_begin,
                    outputs[output].begin() + copy_end,
                    result[output].begin() + start + keep_begin);
        }
      }
    }

    if (config_.normalize_outputs && config_.normalize_before_crop) {
      const float input_rms = Rms(model_audio);
      for (auto& output : result) {
        const float output_rms = Rms(output);
        if (output_rms > 1.0e-12f) {
          const float scale = input_rms / output_rms;
          for (float& value : output) value *= scale;
        }
      }
    }
    for (auto& output : result) output.resize(model_audio.size());
    if (config_.substitute_bandwidth) {
      result[0] = SubstituteBandwidth(input.audio, result[0], config_.sample_rate);
    }
    if (config_.normalize_outputs && !config_.normalize_before_crop) {
      const float input_rms = Rms(model_audio);
      for (auto& output : result) {
        const float output_rms = Rms(output);
        if (output_rms > 1.0e-12f) {
          const float scale = input_rms / output_rms;
          for (float& value : output) value *= scale;
        }
      }
    }
    if (restore_scale != 1.0f) {
      for (auto& output : result) {
        for (float& value : output) value *= restore_scale;
      }
    }
    return result;
  }

 private:
  void ValidateVisual(const ClearVoiceInput& input) const {
    if (config_.visual_frames == 0) return;
    if (input.visual_frames <= 0 || input.visual_height <= 0 ||
        input.visual_width <= 0) {
      throw std::runtime_error("The AV model requires a visual tensor");
    }
    if (input.visual_height != static_cast<int64_t>(config_.visual_height) ||
        input.visual_width != static_cast<int64_t>(config_.visual_width)) {
      throw std::runtime_error("AV visual frames must be 112x112");
    }
    const size_t expected = static_cast<size_t>(input.visual_frames) *
        config_.visual_height * config_.visual_width;
    if (input.visual.size() != expected) {
      throw std::runtime_error("Visual tensor size does not match its shape");
    }
  }

  TensorInput MakeVisualChunk(const ClearVoiceInput& input,
                              size_t audio_start) const {
    TensorInput visual;
    visual.shape = {1, static_cast<int64_t>(config_.visual_frames),
                    static_cast<int64_t>(config_.visual_height),
                    static_cast<int64_t>(config_.visual_width)};
    const size_t frame_size = config_.visual_height * config_.visual_width;
    visual.data.resize(config_.visual_frames * frame_size);
    const size_t first_frame = static_cast<size_t>(std::llround(
        static_cast<double>(audio_start) * 25.0 / config_.sample_rate));
    for (size_t frame = 0; frame < config_.visual_frames; ++frame) {
      const size_t source_frame = std::min(
          first_frame + frame, static_cast<size_t>(input.visual_frames - 1));
      std::copy_n(input.visual.data() + source_frame * frame_size, frame_size,
                  visual.data.data() + frame * frame_size);
    }
    return visual;
  }

  std::vector<std::vector<float>> SplitOutputs(
      const std::vector<TensorOutput>& raw) const {
    if (raw.size() == config_.output_count) {
      std::vector<std::vector<float>> outputs;
      outputs.reserve(raw.size());
      for (const auto& tensor : raw) outputs.push_back(tensor.data);
      return outputs;
    }
    if (raw.size() != 1 || config_.output_count == 1) {
      if (raw.size() == 1 && config_.output_count == 1) {
        return {raw.front().data};
      }
      throw std::runtime_error("Unexpected ONNX output count");
    }
    if (raw.front().data.size() % config_.output_count != 0) {
      throw std::runtime_error("Packed ONNX output cannot be split by source");
    }
    const size_t samples = raw.front().data.size() / config_.output_count;
    std::vector<std::vector<float>> outputs(config_.output_count);
    for (size_t output = 0; output < config_.output_count; ++output) {
      outputs[output].assign(raw.front().data.begin() + output * samples,
                             raw.front().data.begin() + (output + 1) * samples);
    }
    return outputs;
  }

  ModelConfig config_;
  OnnxRunner runner_;
};

ModelConfig ConfigFor(ModelType type) {
  switch (type) {
    case ModelType::kFrcrnSe16k:
      return {16000, 16000, 12000, 1, false, true, true, false, false, 1920000, true,
              true, false, false, 0, 0, 0};
    case ModelType::kMossFormerGanSe16k:
      break;
    case ModelType::kMossFormer2Ss16k:
      return {16000, 32000, 24000, 2, false, true, false, true, false, 0, false, false,
              true, true, 0, 0, 0};
    case ModelType::kMossFormer2Sr48k:
      return {48000, 192000, 144000, 1, false, false, false, false, true, 960000, false,
              false, false, false, 0, 0, 0};
    case ModelType::kAvMossFormer2Tse16k:
      return {16000, 48000, 28800, 1, true, false, false, false, false, 48000, false, false,
              false, false, 75, 112, 112};
    case ModelType::kMossFormer2Se48k:
      break;
  }
  throw std::runtime_error("MossFormer2_SE_48K uses its dedicated runtime");
}

}  // namespace

ModelType ParseModelType(const std::string& value) {
  if (value == "FRCRN_SE_16K") return ModelType::kFrcrnSe16k;
  if (value == "MossFormer2_SE_48K") return ModelType::kMossFormer2Se48k;
  if (value == "MossFormerGAN_SE_16K") {
    return ModelType::kMossFormerGanSe16k;
  }
  if (value == "MossFormer2_SS_16K") return ModelType::kMossFormer2Ss16k;
  if (value == "MossFormer2_SR_48K") return ModelType::kMossFormer2Sr48k;
  if (value == "AV_MossFormer2_TSE_16K") {
    return ModelType::kAvMossFormer2Tse16k;
  }
  throw std::runtime_error("Unknown ClearVoice model type: " + value);
}

std::string ModelTypeName(ModelType type) {
  switch (type) {
    case ModelType::kFrcrnSe16k: return "FRCRN_SE_16K";
    case ModelType::kMossFormer2Se48k: return "MossFormer2_SE_48K";
    case ModelType::kMossFormerGanSe16k: return "MossFormerGAN_SE_16K";
    case ModelType::kMossFormer2Ss16k: return "MossFormer2_SS_16K";
    case ModelType::kMossFormer2Sr48k: return "MossFormer2_SR_48K";
    case ModelType::kAvMossFormer2Tse16k: return "AV_MossFormer2_TSE_16K";
  }
  throw std::runtime_error("Unknown ClearVoice model type");
}

std::unique_ptr<ClearVoice> ClearVoice::Create(ModelType type,
                                                const std::string& model_path,
                                                const std::string& provider) {
  if (type == ModelType::kMossFormer2Se48k) {
    return std::make_unique<MossFormer2SeModel>(model_path, provider);
  }
  if (type == ModelType::kMossFormerGanSe16k) {
    return std::make_unique<MossFormerGanModel>(model_path, provider);
  }
  return std::make_unique<DirectOnnxModel>(ConfigFor(type), model_path,
                                           provider);
}
