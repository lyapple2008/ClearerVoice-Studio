#pragma once

#include <memory>
#include <string>
#include <vector>

#include "clearvoice.h"

class MossFormerGanModel final : public ClearVoice {
 public:
  MossFormerGanModel(const std::string& model_path,
                     const std::string& provider);
  ~MossFormerGanModel() override;

  int32_t SampleRate() const override;
  size_t OutputCount() const override;
  std::vector<std::vector<float>> Process(
      const ClearVoiceInput& input) const override;
  const std::string& ActiveProvider() const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
