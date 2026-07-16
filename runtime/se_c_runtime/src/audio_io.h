#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct AudioData {
  int32_t sample_rate = 0;
  std::vector<std::vector<float>> channels;
};

AudioData ReadWav(const std::string& path);
void WriteWav16(const std::string& path, const AudioData& audio);
