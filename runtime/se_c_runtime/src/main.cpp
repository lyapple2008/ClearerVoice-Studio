#include <exception>
#include <iostream>
#include <string>

#include "audio_io.h"
#include "enhancer.h"

int main(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    std::cerr << "Usage: " << argv[0]
              << " MODEL.onnx INPUT.wav OUTPUT.wav [--provider auto|cpu|coreml|cuda]\n";
    return 2;
  }

  std::string provider = "auto";
  if (argc == 6) {
    if (std::string(argv[4]) != "--provider") {
      std::cerr << "Expected --provider before provider name\n";
      return 2;
    }
    provider = argv[5];
  }

  try {
    AudioData audio = ReadWav(argv[2]);
    if (audio.sample_rate != 48000) {
      throw std::runtime_error("Input WAV must use a 48000 Hz sample rate");
    }
    Enhancer enhancer(argv[1], provider);
    std::cout << "Execution provider: " << enhancer.ActiveProvider() << '\n';
    for (auto& channel : audio.channels) channel = enhancer.Process(channel);
    WriteWav16(argv[3], audio);
    std::cout << "Wrote " << argv[3] << '\n';
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
