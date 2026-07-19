#include "bandwidth_sub.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <kiss_fftr.h>

namespace {

constexpr int kOrder = 4;
constexpr int kStftSize = 256;
constexpr int kStftHop = 128;
constexpr double kPi = 3.14159265358979323846;

using Complex = std::complex<double>;

std::vector<Complex> PolynomialFromRoots(const std::vector<Complex>& roots) {
  std::vector<Complex> coefficients{1.0};
  for (const Complex& root : roots) {
    std::vector<Complex> next(coefficients.size() + 1);
    for (size_t i = 0; i < coefficients.size(); ++i) {
      next[i] += coefficients[i];
      next[i + 1] -= coefficients[i] * root;
    }
    coefficients = std::move(next);
  }
  return coefficients;
}

std::pair<std::vector<double>, std::vector<double>> Butterworth(
    double cutoff_hz, double sample_rate, bool highpass) {
  const double normalized = cutoff_hz / (sample_rate * 0.5);
  if (!(normalized > 0.0 && normalized < 1.0)) {
    throw std::runtime_error("Detected bandwidth is outside filter range");
  }
  const double fs = 2.0;
  const double warped = 2.0 * fs * std::tan(kPi * normalized / fs);

  std::vector<Complex> prototype_poles;
  for (int index = 0; index < kOrder; ++index) {
    const double angle = kPi * (2.0 * index + 1.0 + kOrder) /
        (2.0 * kOrder);
    prototype_poles.emplace_back(std::cos(angle), std::sin(angle));
  }

  std::vector<Complex> analog_zeros;
  std::vector<Complex> analog_poles;
  double analog_gain = 1.0;
  if (highpass) {
    analog_zeros.assign(kOrder, Complex{0.0, 0.0});
    for (const auto& pole : prototype_poles) {
      analog_poles.push_back(warped / pole);
    }
  } else {
    for (const auto& pole : prototype_poles) {
      analog_poles.push_back(warped * pole);
    }
    analog_gain = std::pow(warped, kOrder);
  }

  const double fs2 = 2.0 * fs;
  std::vector<Complex> digital_zeros;
  std::vector<Complex> digital_poles;
  for (const auto& zero : analog_zeros) {
    digital_zeros.push_back((fs2 + zero) / (fs2 - zero));
  }
  for (const auto& pole : analog_poles) {
    digital_poles.push_back((fs2 + pole) / (fs2 - pole));
  }
  while (digital_zeros.size() < digital_poles.size()) {
    digital_zeros.emplace_back(-1.0, 0.0);
  }

  Complex numerator_gain = analog_gain;
  for (const auto& zero : analog_zeros) numerator_gain *= fs2 - zero;
  Complex denominator_gain = 1.0;
  for (const auto& pole : analog_poles) denominator_gain *= fs2 - pole;
  const double digital_gain = (numerator_gain / denominator_gain).real();

  auto b_complex = PolynomialFromRoots(digital_zeros);
  auto a_complex = PolynomialFromRoots(digital_poles);
  std::vector<double> b(b_complex.size());
  std::vector<double> a(a_complex.size());
  for (size_t i = 0; i < b.size(); ++i) b[i] = b_complex[i].real() * digital_gain;
  for (size_t i = 0; i < a.size(); ++i) a[i] = a_complex[i].real();
  const double a0 = a.front();
  for (double& value : a) value /= a0;
  for (double& value : b) value /= a0;
  return {b, a};
}

std::vector<double> Filter(const std::vector<double>& input,
                           const std::vector<double>& b,
                           const std::vector<double>& a) {
  std::vector<double> output(input.size());
  std::vector<double> state(std::max(a.size(), b.size()) - 1);
  auto step = [&](double value) {
    double result = b[0] * value + state[0];
    for (size_t i = 1; i < state.size(); ++i) {
      const double bi = i < b.size() ? b[i] : 0.0;
      const double ai = i < a.size() ? a[i] : 0.0;
      state[i - 1] = bi * value + state[i] - ai * result;
    }
    const size_t last = state.size();
    const double bi = last < b.size() ? b[last] : 0.0;
    const double ai = last < a.size() ? a[last] : 0.0;
    state[last - 1] = bi * value - ai * result;
    return result;
  };
  for (int i = 0; i < 256; ++i) step(input.front());
  for (size_t i = 0; i < input.size(); ++i) output[i] = step(input[i]);
  return output;
}

std::vector<float> ZeroPhaseFilter(const std::vector<float>& input,
                                   double cutoff_hz, int32_t sample_rate,
                                   bool highpass) {
  if (input.size() <= 16) return input;
  const auto [b, a] = Butterworth(cutoff_hz, sample_rate, highpass);
  constexpr size_t pad = 15;
  std::vector<double> extended(input.size() + 2 * pad);
  for (size_t i = 0; i < pad; ++i) {
    extended[pad - 1 - i] = 2.0 * input.front() - input[i + 1];
    extended[pad + input.size() + i] =
        2.0 * input.back() - input[input.size() - 2 - i];
  }
  std::copy(input.begin(), input.end(), extended.begin() + pad);
  auto forward = Filter(extended, b, a);
  std::reverse(forward.begin(), forward.end());
  auto backward = Filter(forward, b, a);
  std::reverse(backward.begin(), backward.end());
  std::vector<float> output(input.size());
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = static_cast<float>(backward[i + pad]);
  }
  return output;
}

double DetectHighBandwidth(const std::vector<float>& signal,
                           int32_t sample_rate) {
  const size_t frames = signal.empty()
      ? 0
      : 1 + (signal.size() + kStftHop - 1) / kStftHop;
  std::vector<double> energy(kStftSize / 2 + 1);
  std::vector<float> window(kStftSize);
  for (int i = 0; i < kStftSize; ++i) {
    window[i] = static_cast<float>(
        0.5 - 0.5 * std::cos(2.0 * kPi * i / (kStftSize - 1)));
  }
  kiss_fftr_cfg config = kiss_fftr_alloc(kStftSize, 0, nullptr, nullptr);
  if (!config) throw std::runtime_error("Failed to allocate bandwidth FFT");
  std::vector<float> frame(kStftSize);
  std::vector<kiss_fft_cpx> bins(kStftSize / 2 + 1);
  for (size_t index = 0; index < frames; ++index) {
    const int64_t start = static_cast<int64_t>(index * kStftHop) - kStftSize / 2;
    for (int i = 0; i < kStftSize; ++i) {
      const int64_t source = start + i;
      frame[i] = source >= 0 && source < static_cast<int64_t>(signal.size())
          ? signal[static_cast<size_t>(source)] * window[i]
          : 0.0f;
    }
    kiss_fftr(config, frame.data(), bins.data());
    for (size_t bin = 0; bin < energy.size(); ++bin) {
      energy[bin] += static_cast<double>(bins[bin].r) * bins[bin].r +
                     static_cast<double>(bins[bin].i) * bins[bin].i;
    }
  }
  kiss_fft_free(config);
  const double total = std::accumulate(energy.begin(), energy.end(), 0.0);
  if (total <= 0.0) return sample_rate * 0.25;
  double cumulative = 0.0;
  for (size_t bin = 0; bin < energy.size(); ++bin) {
    cumulative += energy[bin];
    if (cumulative / total >= 0.9996) {
      return static_cast<double>(bin) * sample_rate / kStftSize;
    }
  }
  return sample_rate * 0.5 - sample_rate / kStftSize;
}

}  // namespace

std::vector<float> SubstituteBandwidth(const std::vector<float>& original,
                                       const std::vector<float>& generated,
                                       int32_t sample_rate) {
  const size_t size = std::min(original.size(), generated.size());
  if (size == 0) return {};
  std::vector<float> original_trimmed(original.begin(), original.begin() + size);
  std::vector<float> generated_trimmed(generated.begin(), generated.begin() + size);
  double cutoff = DetectHighBandwidth(original_trimmed, sample_rate);
  cutoff = std::clamp(cutoff, 1.0, sample_rate * 0.5 - 1.0);
  const auto low = ZeroPhaseFilter(original_trimmed, cutoff, sample_rate, false);
  const auto high = ZeroPhaseFilter(generated_trimmed, cutoff, sample_rate, true);
  std::vector<float> output(size);
  const size_t fade_samples = std::min<size_t>(size, sample_rate / 10);
  for (size_t i = 0; i < size; ++i) {
    const float substituted = low[i] + high[i];
    const float fade = i < fade_samples && fade_samples > 1
        ? static_cast<float>(i) / static_cast<float>(fade_samples - 1)
        : 1.0f;
    output[i] = (1.0f - fade) * original_trimmed[i] + fade * substituted;
  }
  return output;
}
