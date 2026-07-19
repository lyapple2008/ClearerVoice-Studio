#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct TensorInput {
  std::vector<float> data;
  std::vector<int64_t> shape;
};

struct TensorOutput {
  std::vector<float> data;
  std::vector<int64_t> shape;
};

class OnnxRunner {
 public:
  OnnxRunner(const std::string& model_path, const std::string& provider);
  ~OnnxRunner();

  std::vector<TensorOutput> Run(const std::vector<TensorInput>& inputs) const;
  size_t InputCount() const;
  size_t OutputCount() const;
  const std::string& ActiveProvider() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
