#include "resample.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kRadius = 16;

double Sinc(double value) {
  if (std::abs(value) < 1.0e-12) return 1.0;
  const double angle = kPi * value;
  return std::sin(angle) / angle;
}

}  // namespace

std::vector<float> Resample(const std::vector<float>& input,
                            int32_t input_rate,
                            int32_t output_rate) {
  if (input_rate <= 0 || output_rate <= 0) {
    throw std::runtime_error("Sample rates must be positive");
  }
  if (input.empty() || input_rate == output_rate) return input;

  const size_t output_size = static_cast<size_t>(std::llround(
      static_cast<double>(input.size()) * output_rate / input_rate));
  std::vector<float> output(output_size);
  const double ratio = static_cast<double>(input_rate) / output_rate;
  const double cutoff = std::min(1.0, static_cast<double>(output_rate) /
                                          static_cast<double>(input_rate));

  for (size_t index = 0; index < output_size; ++index) {
    const double source = static_cast<double>(index) * ratio;
    const int64_t center = static_cast<int64_t>(std::floor(source));
    double sum = 0.0;
    double weight_sum = 0.0;
    for (int offset = -kRadius + 1; offset <= kRadius; ++offset) {
      const int64_t source_index = center + offset;
      if (source_index < 0 ||
          source_index >= static_cast<int64_t>(input.size())) {
        continue;
      }
      const double distance = source - static_cast<double>(source_index);
      const double window = 0.5 + 0.5 *
          std::cos(kPi * distance / static_cast<double>(kRadius));
      const double weight = cutoff * Sinc(distance * cutoff) * window;
      sum += static_cast<double>(input[source_index]) * weight;
      weight_sum += weight;
    }
    output[index] = weight_sum == 0.0
        ? 0.0f
        : static_cast<float>(sum / weight_sum);
  }
  return output;
}
