// 极简 ONNX Runtime 封装：只支持单输入单输出的场景（切分/姿态都够用）。
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace ed {

class OrtModel {
 public:
  OrtModel(const std::string& model_path, int threads = 4);

  bool ready() const { return session_ != nullptr; }
  const std::vector<int64_t>& input_shape() const { return input_shape_; }
  const std::vector<int64_t>& output_shape() const { return output_shape_; }

  // float32 输入（NCHW/任意 shape），返回第一个输出
  std::vector<float> run(const std::vector<float>& input, std::vector<int64_t>* out_shape);
  // uint8 输入（TransNetV2 需要）
  std::vector<float> run_u8(const std::vector<uint8_t>& input, std::vector<int64_t>* out_shape);

 private:
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "auto_editor"};
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<const char*> out_names_;
};

}  // namespace ed
