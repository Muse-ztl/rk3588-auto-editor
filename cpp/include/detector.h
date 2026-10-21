// YOLOv8 RKNN 检测器：letterbox -> NPU -> DFL 解码 -> 分类 NMS。
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <rknn_api.h>

#include "types.h"

namespace ed {

// 分阶段计时，用来回答“慢在 preprocess、NPU 还是后处理”。
struct DetectTiming {
  double preprocess_ms = 0.0;
  double input_ms = 0.0;
  double run_ms = 0.0;
  double output_ms = 0.0;
  double decode_ms = 0.0;
  double nms_ms = 0.0;
  double total_ms = 0.0;
};

class YoloV8 {
 public:
  YoloV8(const std::string& model_path, float conf = 0.30f, float iou = 0.45f,
         int imgsz = 640, int core_mask = 0, int max_det = 100, bool zero_copy = true);
  ~YoloV8();

  YoloV8(const YoloV8&) = delete;
  YoloV8& operator=(const YoloV8&) = delete;

  std::vector<Box> detect(const cv::Mat& bgr);
  double last_ms() const { return last_ms_; }
  const DetectTiming& timing() const { return timing_; }
  bool zero_copy() const { return zero_copy_; }

 private:
  cv::Mat make_rgb(const cv::Mat& bgr, float& ratio, float& pad_x, float& pad_y) const;
  bool setup_zero_copy();
  void release_mem();
  std::vector<Box> nms(std::vector<Box>& cands, float ratio, float pad_x, float pad_y) const;

  rknn_context ctx_ = 0;
  bool alive_ = false;
  rknn_input_output_num io_num_{};
  rknn_tensor_attr out_attr_[12]{};
  rknn_tensor_attr in_attr_{};
  rknn_tensor_mem* in_mem_ = nullptr;
  std::vector<rknn_tensor_mem*> out_mem_;
  std::vector<cv::Mat> out_f32_;
  bool zero_copy_ = false;
  int n_out_ = 0;
  int imgsz_ = 640;
  float conf_ = 0.30f;
  float iou_ = 0.45f;
  int max_det_ = 100;
  double last_ms_ = 0.0;
  DetectTiming timing_;
};

}  // namespace ed
