// 人脸：YuNet（ONNX Runtime 直接推理 + 自解码）。
//
// 不用 cv::FaceDetectorYN：板上的 OpenCV 4.5.4 加载 2023mar 权重会报
// "Layer with requested id=-1"，YuNet 的 DNN 图需要 4.6+ 才支持。
// 这里按 libfacedetection 的解码规则自己处理多尺度输出。
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace ed {

struct Face {
  cv::Rect2f box;
  float score = 0.f;
  cv::Point2f landmarks[5];  // 右眼、左眼、鼻尖、右嘴角、左嘴角
  float frontal = 0.f;       // 0~1，越接近正脸越高
};

class FaceDetector {
 public:
  explicit FaceDetector(const std::string& model_path, float score_thr = 0.6f);
  bool ready() const;

  // 在给定人体框内找人脸（先裁人再缩放，比整帧检测更稳，也更快）
  std::vector<Face> detect_in_box(const cv::Mat& frame, const cv::Rect2f& person_box);

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  float score_thr_;
};

}  // namespace ed
