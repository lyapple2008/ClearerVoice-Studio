#include "audio_io.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

uint16_t ReadU16(std::istream& stream) {
  uint8_t bytes[2];
  stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  return static_cast<uint16_t>(bytes[0]) |
         (static_cast<uint16_t>(bytes[1]) << 8);
}

uint32_t ReadU32(std::istream& stream) {
  uint8_t bytes[4];
  stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

void WriteU16(std::ostream& stream, uint16_t value) {
  const uint8_t bytes[] = {static_cast<uint8_t>(value),
                           static_cast<uint8_t>(value >> 8)};
  stream.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void WriteU32(std::ostream& stream, uint32_t value) {
  const uint8_t bytes[] = {
      static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
      static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24)};
  stream.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

int32_t ReadSigned24(const uint8_t* bytes) {
  int32_t value = static_cast<int32_t>(bytes[0]) |
                  (static_cast<int32_t>(bytes[1]) << 8) |
                  (static_cast<int32_t>(bytes[2]) << 16);
  if (value & 0x00800000) value |= static_cast<int32_t>(0xFF000000);
  return value;
}

}  // namespace

AudioData ReadWav(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Cannot open input WAV: " + path);

  char riff[4];
  stream.read(riff, 4);
  ReadU32(stream);
  char wave[4];
  stream.read(wave, 4);
  if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
    throw std::runtime_error("Only little-endian RIFF/WAVE files are supported");
  }

  uint16_t format = 0;
  uint16_t channel_count = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  std::vector<uint8_t> data;

  while (stream && (!format || data.empty())) {
    char id[4];
    stream.read(id, 4);
    if (!stream) break;
    const uint32_t size = ReadU32(stream);
    if (std::memcmp(id, "fmt ", 4) == 0) {
      format = ReadU16(stream);
      channel_count = ReadU16(stream);
      sample_rate = ReadU32(stream);
      ReadU32(stream);
      ReadU16(stream);
      bits_per_sample = ReadU16(stream);
      if (size > 16) stream.seekg(size - 16, std::ios::cur);
    } else if (std::memcmp(id, "data", 4) == 0) {
      data.resize(size);
      stream.read(reinterpret_cast<char*>(data.data()), size);
    } else {
      stream.seekg(size, std::ios::cur);
    }
    if (size & 1U) stream.seekg(1, std::ios::cur);
  }

  if ((format != 1 && format != 3) || channel_count == 0 || data.empty()) {
    throw std::runtime_error("Unsupported or incomplete WAV file");
  }
  if (format == 1 && bits_per_sample != 16 && bits_per_sample != 24 &&
      bits_per_sample != 32) {
    throw std::runtime_error("PCM WAV must use 16, 24, or 32 bits per sample");
  }
  if (format == 3 && bits_per_sample != 32) {
    throw std::runtime_error("Float WAV must use 32-bit samples");
  }

  const size_t bytes_per_sample = bits_per_sample / 8;
  const size_t frame_count = data.size() / (bytes_per_sample * channel_count);
  AudioData audio;
  audio.sample_rate = static_cast<int32_t>(sample_rate);
  audio.channels.assign(channel_count, std::vector<float>(frame_count));

  for (size_t frame = 0; frame < frame_count; ++frame) {
    for (size_t channel = 0; channel < channel_count; ++channel) {
      const uint8_t* sample = data.data() +
          (frame * channel_count + channel) * bytes_per_sample;
      float value = 0.0f;
      if (format == 3) {
        std::memcpy(&value, sample, sizeof(value));
      } else if (bits_per_sample == 16) {
        const int16_t pcm = static_cast<int16_t>(
            static_cast<uint16_t>(sample[0]) |
            (static_cast<uint16_t>(sample[1]) << 8));
        value = static_cast<float>(pcm) / 32768.0f;
      } else if (bits_per_sample == 24) {
        value = static_cast<float>(ReadSigned24(sample)) / 8388608.0f;
      } else {
        int32_t pcm;
        std::memcpy(&pcm, sample, sizeof(pcm));
        value = static_cast<float>(pcm) / 2147483648.0f;
      }
      audio.channels[channel][frame] = value;
    }
  }
  return audio;
}

void WriteWav16(const std::string& path, const AudioData& audio) {
  if (audio.channels.empty()) throw std::runtime_error("No audio to write");
  const size_t frame_count = audio.channels.front().size();
  for (const auto& channel : audio.channels) {
    if (channel.size() != frame_count) {
      throw std::runtime_error("Output channels have different lengths");
    }
  }

  std::ofstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Cannot open output WAV: " + path);
  const uint16_t channel_count = static_cast<uint16_t>(audio.channels.size());
  const uint32_t data_size = static_cast<uint32_t>(
      frame_count * channel_count * sizeof(int16_t));

  stream.write("RIFF", 4);
  WriteU32(stream, 36 + data_size);
  stream.write("WAVEfmt ", 8);
  WriteU32(stream, 16);
  WriteU16(stream, 1);
  WriteU16(stream, channel_count);
  WriteU32(stream, static_cast<uint32_t>(audio.sample_rate));
  WriteU32(stream, static_cast<uint32_t>(audio.sample_rate) * channel_count * 2);
  WriteU16(stream, channel_count * 2);
  WriteU16(stream, 16);
  stream.write("data", 4);
  WriteU32(stream, data_size);

  for (size_t frame = 0; frame < frame_count; ++frame) {
    for (const auto& channel : audio.channels) {
      const float clipped = std::clamp(channel[frame], -1.0f, 32767.0f / 32768.0f);
      const int16_t pcm = static_cast<int16_t>(std::floor(clipped * 32768.0f));
      WriteU16(stream, static_cast<uint16_t>(pcm));
    }
  }
}
