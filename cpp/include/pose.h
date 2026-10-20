// 姿态：YOLOv8n-pose（17 关键点），用于判断构图与"人物是否完整入画"。
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "ort_model.h"

namespace ed {

struct PoseResult {
  bool ok = false;
  float score = 0.f;
  float framing = 0.f;   // 0~1，头部位置 + 关键点完整度
  cv::Point2f kpts[17];
  float conf[17] = {0.f};
};

class PoseEstimator {
 public:
  explicit PoseEstimator(const std::string& model_path, int threads = 4);
  bool ready() const { return model_ && model_->ready(); }

  // person_box：原图坐标下的人体框，用来在多个人里选出对应那个
  PoseResult estimate(const cv::Mat& frame, const cv::Rect2f& person_box);

 private:
  std::unique_ptr<OrtModel> model_;
  int imgsz_ = 640;
};

}  // namespace ed
