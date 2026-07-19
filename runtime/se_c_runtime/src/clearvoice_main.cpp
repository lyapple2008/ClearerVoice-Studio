#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "audio_io.h"
#include "clearvoice.h"
#include "resample.h"

namespace {

std::vector<float> ReadFloatFile(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("Cannot open visual tensor: " + path);
  const std::streamsize bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
    throw std::runtime_error("Visual tensor must contain raw float32 values");
  }
  stream.seekg(0);
  std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
  stream.read(reinterpret_cast<char*>(values.data()), bytes);
  if (!stream) throw std::runtime_error("Cannot read visual tensor: " + path);
  return values;
}

std::string SourcePath(const std::string& path, size_t source) {
  const std::filesystem::path original(path);
  const std::string suffix = "_s" + std::to_string(source + 1);
  return (original.parent_path() /
          (original.stem().string() + suffix + original.extension().string()))
      .string();
}

void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " MODEL_TYPE MODEL.onnx INPUT.wav OUTPUT.wav"
         " [--provider auto|cpu|coreml|cuda] [--visual frames.f32]\n"
      << "MODEL_TYPE: FRCRN_SE_16K, MossFormer2_SE_48K, "
         "MossFormerGAN_SE_16K, MossFormer2_SS_16K, "
         "MossFormer2_SR_48K, AV_MossFormer2_TSE_16K\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    PrintUsage(argv[0]);
    return 2;
  }

  std::string provider = "auto";
  std::string visual_path;
  for (int index = 5; index < argc; index += 2) {
    if (index + 1 >= argc) {
      PrintUsage(argv[0]);
      return 2;
    }
    const std::string option = argv[index];
    if (option == "--provider") {
      provider = argv[index + 1];
    } else if (option == "--visual") {
      visual_path = argv[index + 1];
    } else {
      std::cerr << "Unknown option: " << option << '\n';
      return 2;
    }
  }

  try {
    const ModelType type = ParseModelType(argv[1]);
    auto model = ClearVoice::Create(type, argv[2], provider);
    AudioData input = ReadWav(argv[3]);
    if (type == ModelType::kAvMossFormer2Tse16k &&
        input.channels.size() != 1) {
      throw std::runtime_error("The AV model requires mono audio");
    }

    ClearVoiceInput common_input;
    if (!visual_path.empty()) {
      common_input.visual = ReadFloatFile(visual_path);
      constexpr size_t kFrameSize = 112 * 112;
      if (common_input.visual.size() % kFrameSize != 0) {
        throw std::runtime_error(
            "Visual tensor size must be a multiple of 112x112");
      }
      common_input.visual_frames = static_cast<int64_t>(
          common_input.visual.size() / kFrameSize);
      common_input.visual_height = 112;
      common_input.visual_width = 112;
    }

    std::vector<AudioData> outputs(model->OutputCount());
    const int32_t output_rate = type == ModelType::kMossFormer2Sr48k
        ? model->SampleRate()
        : input.sample_rate;
    for (auto& output : outputs) {
      output.sample_rate = output_rate;
      output.channels.resize(input.channels.size());
    }

    for (size_t channel = 0; channel < input.channels.size(); ++channel) {
      common_input.audio = Resample(input.channels[channel], input.sample_rate,
                                    model->SampleRate());
      auto channel_outputs = model->Process(common_input);
      if (channel_outputs.size() != outputs.size()) {
        throw std::runtime_error("Model returned an unexpected source count");
      }
      for (size_t source = 0; source < outputs.size(); ++source) {
        outputs[source].channels[channel] = Resample(
            channel_outputs[source], model->SampleRate(), output_rate);
        if (output_rate == input.sample_rate) {
          outputs[source].channels[channel].resize(input.channels[channel].size());
        }
      }
    }

    std::cout << "Model: " << ModelTypeName(type) << '\n';
    std::cout << "Execution provider: " << model->ActiveProvider() << '\n';
    for (size_t source = 0; source < outputs.size(); ++source) {
      const std::string output_path = outputs.size() == 1
          ? argv[4]
          : SourcePath(argv[4], source);
      WriteWav16(output_path, outputs[source]);
      std::cout << "Wrote " << output_path << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
