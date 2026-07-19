#pragma once

#include <cstdint>
#include <vector>

std::vector<float> Resample(const std::vector<float>& input,
                            int32_t input_rate,
                            int32_t output_rate);
