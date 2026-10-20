#include "ort_model.h"

#include <stdexcept>

namespace ed {

OrtModel::OrtModel(const std::string& model_path, int threads) {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(threads);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  try {
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
  } catch (const Ort::Exception&) {
    session_.reset();  // 模型缺失时允许上层降级
    return;
  }

  Ort::AllocatorWithDefaultOptions alloc;
  input_name_ = session_->GetInputNameAllocated(0, alloc).get();
  output_name_ = session_->GetOutputNameAllocated(0, alloc).get();
  input_shape_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  output_shape_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  out_names_.push_back(output_name_.c_str());
}

std::vector<float> OrtModel::run(const std::vector<float>& input,
                                 std::vector<int64_t>* out_shape) {
  if (!session_) throw std::runtime_error("ort model not loaded");
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(input.data()),
                                                input.size(), input_shape_.data(),
                                                input_shape_.size());
  const char* in_name = input_name_.c_str();
  const char* out_name = output_name_.c_str();
  auto outs = session_->Run(Ort::RunOptions{nullptr}, &in_name, &tensor, 1, &out_name, 1);
  const auto& info = outs[0].GetTensorTypeAndShapeInfo();
  if (out_shape) *out_shape = info.GetShape();
  const float* p = outs[0].GetTensorData<float>();
  return std::vector<float>(p, p + info.GetElementCount());
}

std::vector<float> OrtModel::run_u8(const std::vector<uint8_t>& input,
                                    std::vector<int64_t>* out_shape) {
  if (!session_) throw std::runtime_error("ort model not loaded");
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<uint8_t>(mem, const_cast<uint8_t*>(input.data()),
                                                  input.size(), input_shape_.data(),
                                                  input_shape_.size());
  const char* in_name = input_name_.c_str();
  const char* out_name = output_name_.c_str();
  auto outs = session_->Run(Ort::RunOptions{nullptr}, &in_name, &tensor, 1, &out_name, 1);
  const auto& info = outs[0].GetTensorTypeAndShapeInfo();
  if (out_shape) *out_shape = info.GetShape();
  const float* p = outs[0].GetTensorData<float>();
  return std::vector<float>(p, p + info.GetElementCount());
}

}  // namespace ed
