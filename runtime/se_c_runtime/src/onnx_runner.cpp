#include "onnx_runner.h"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <onnxruntime_cxx_api.h>

namespace {

size_t ElementCount(const std::vector<int64_t>& shape) {
  if (shape.empty()) return 0;
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         [](size_t product, int64_t dimension) {
                           if (dimension <= 0) {
                             throw std::runtime_error(
                                 "Tensor shapes must be fully specified and positive");
                           }
                           return product * static_cast<size_t>(dimension);
                         });
}

void ConfigureProvider(Ort::SessionOptions& options,
                       const std::string& requested,
                       std::string& active_provider) {
  if (requested != "auto" && requested != "cpu" && requested != "coreml" &&
      requested != "cuda") {
    throw std::runtime_error("Provider must be auto, cpu, coreml, or cuda");
  }
  if (requested == "cpu") {
    active_provider = "cpu";
    return;
  }

#ifdef __APPLE__
  if (requested == "auto" || requested == "coreml") {
    try {
      options.AppendExecutionProvider(
          "CoreML", {{"ModelFormat", "MLProgram"}});
      active_provider = "coreml";
      return;
    } catch (const Ort::Exception&) {
      if (requested == "coreml") throw;
    }
  }
#endif

  if (requested == "auto" || requested == "cuda") {
    try {
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = 0;
      options.AppendExecutionProvider_CUDA(cuda_options);
      active_provider = "cuda";
      return;
    } catch (const Ort::Exception&) {
      if (requested == "cuda") throw;
    }
  }
  active_provider = "cpu";
}

}  // namespace

class OnnxRunner::Impl {
 public:
  Impl(const std::string& model_path, const std::string& requested_provider) {
    const OrtApiBase* api_base = OrtGetApiBase();
    if (api_base == nullptr || api_base->GetApi(ORT_API_VERSION) == nullptr) {
      throw std::runtime_error("ONNX Runtime headers and library are incompatible");
    }

    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                     "clearvoice_runtime");
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    ConfigureProvider(options, requested_provider, active_provider);
    const std::filesystem::path path(model_path);
    try {
      session = std::make_unique<Ort::Session>(*env, path.c_str(), options);
    } catch (const std::exception&) {
      if (requested_provider != "auto" || active_provider == "cpu") throw;
      options = Ort::SessionOptions{};
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      active_provider = "cpu";
      session = std::make_unique<Ort::Session>(*env, path.c_str(), options);
    }

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session->GetInputCount(); ++i) {
      auto name = session->GetInputNameAllocated(i, allocator);
      input_names_storage.emplace_back(name.get());
    }
    for (size_t i = 0; i < session->GetOutputCount(); ++i) {
      auto name = session->GetOutputNameAllocated(i, allocator);
      output_names_storage.emplace_back(name.get());
    }
    for (const auto& name : input_names_storage) input_names.push_back(name.c_str());
    for (const auto& name : output_names_storage) output_names.push_back(name.c_str());
  }

  std::vector<TensorOutput> Run(const std::vector<TensorInput>& inputs) const {
    if (inputs.size() != input_names.size()) {
      throw std::runtime_error("ONNX model expects " +
                               std::to_string(input_names.size()) +
                               " inputs, got " + std::to_string(inputs.size()));
    }
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> values;
    values.reserve(inputs.size());
    for (const auto& input : inputs) {
      if (ElementCount(input.shape) != input.data.size()) {
        throw std::runtime_error("Input tensor data size does not match shape");
      }
      values.emplace_back(Ort::Value::CreateTensor<float>(
          memory, const_cast<float*>(input.data.data()), input.data.size(),
          input.shape.data(), input.shape.size()));
    }

    auto ort_outputs = session->Run(
        Ort::RunOptions{nullptr}, input_names.data(), values.data(), values.size(),
        output_names.data(), output_names.size());
    std::vector<TensorOutput> outputs;
    outputs.reserve(ort_outputs.size());
    for (const auto& value : ort_outputs) {
      const auto info = value.GetTensorTypeAndShapeInfo();
      TensorOutput output;
      output.shape = info.GetShape();
      const size_t count = info.GetElementCount();
      const float* data = value.GetTensorData<float>();
      output.data.assign(data, data + count);
      outputs.push_back(std::move(output));
    }
    return outputs;
  }

  std::unique_ptr<Ort::Env> env;
  Ort::SessionOptions options;
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> input_names_storage;
  std::vector<std::string> output_names_storage;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  std::string active_provider = "cpu";
};

OnnxRunner::OnnxRunner(const std::string& model_path,
                       const std::string& provider)
    : impl_(std::make_unique<Impl>(model_path, provider)) {}

OnnxRunner::~OnnxRunner() = default;

std::vector<TensorOutput> OnnxRunner::Run(
    const std::vector<TensorInput>& inputs) const {
  return impl_->Run(inputs);
}

size_t OnnxRunner::InputCount() const { return impl_->input_names.size(); }

size_t OnnxRunner::OutputCount() const { return impl_->output_names.size(); }

const std::string& OnnxRunner::ActiveProvider() const {
  return impl_->active_provider;
}
