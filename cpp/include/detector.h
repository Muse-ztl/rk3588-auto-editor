// YOLOv8 RKNN 检测器：letterbox -> NPU -> DFL 解码 -> 分类 NMS。
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <rknn_api.h>

#include "types.h"

namespace ed {

class YoloV8 {
 public:
  YoloV8(const std::string& model_path, float conf = 0.30f, float iou = 0.45f,
         int imgsz = 640, int core_mask = 0, int max_det = 100);
  ~YoloV8();

  YoloV8(const YoloV8&) = delete;
  YoloV8& operator=(const YoloV8&) = delete;

  std::vector<Box> detect(const cv::Mat& bgr);
  double last_ms() const { return last_ms_; }

 private:
  void preprocess(const cv::Mat& bgr, std::vector<float>& out, float& ratio,
                  float& pad_x, float& pad_y) const;
  // RKNN 输出布局不确定（不同版本可能给 NCHW 或 NHWC），按 dims 自适应取数
  static float at(const float* p, int c, int y, int x, int H, int W, int C, bool chw);
  void decode_scale(const std::vector<float>& box, const float* cls, int H, int W,
                    int nc, float stride, std::vector<Box>& cands) const;
  std::vector<Box> nms(std::vector<Box>& cands, float ratio, float pad_x, float pad_y) const;

  rknn_context ctx_ = 0;
  bool alive_ = false;
  rknn_input_output_num io_num_{};
  rknn_tensor_attr out_attr_[12]{};
  int n_out_ = 0;
  int imgsz_ = 640;
  float conf_ = 0.30f;
  float iou_ = 0.45f;
  int max_det_ = 100;
  double last_ms_ = 0.0;
};

}  // namespace ed
