#pragma once

#include <memory>
#include <string>
#include <vector>

class Enhancer {
 public:
  Enhancer(const std::string& model_path, const std::string& provider);
  ~Enhancer();
  std::vector<float> Process(const std::vector<float>& audio) const;
  const std::string& ActiveProvider() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
