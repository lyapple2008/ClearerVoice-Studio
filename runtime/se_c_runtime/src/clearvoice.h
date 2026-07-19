#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class ModelType {
  kFrcrnSe16k,
  kMossFormer2Se48k,
  kMossFormerGanSe16k,
  kMossFormer2Ss16k,
  kMossFormer2Sr48k,
  kAvMossFormer2Tse16k,
};

ModelType ParseModelType(const std::string& value);
std::string ModelTypeName(ModelType type);

struct ClearVoiceInput {
  std::vector<float> audio;
  std::vector<float> visual;
  int64_t visual_frames = 0;
  int64_t visual_height = 0;
  int64_t visual_width = 0;
};

class ClearVoice {
 public:
  virtual ~ClearVoice() = default;

  static std::unique_ptr<ClearVoice> Create(ModelType type,
                                             const std::string& model_path,
                                             const std::string& provider);

  virtual int32_t SampleRate() const = 0;
  virtual size_t OutputCount() const = 0;
  virtual std::vector<std::vector<float>> Process(
      const ClearVoiceInput& input) const = 0;
  virtual const std::string& ActiveProvider() const = 0;
};
