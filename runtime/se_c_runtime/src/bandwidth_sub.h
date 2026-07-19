#pragma once

#include <cstdint>
#include <vector>

std::vector<float> SubstituteBandwidth(const std::vector<float>& original,
                                       const std::vector<float>& generated,
                                       int32_t sample_rate);
