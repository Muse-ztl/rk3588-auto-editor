// 外观特征（HSV 直方图）与画面质量打分。
#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "types.h"

namespace ed {

// 每帧每个框一个 1x32 的 L2 归一化特征（H16 + S8 + V8）
cv::Mat hsv_features(const cv::Mat& frame, const std::vector<Box>& boxes);

struct Aesthetic {
  double sharpness = 0.0;
  double sharpness_n = 0.0;
  double colorfulness = 0.0;
  double colorfulness_n = 0.0;
  double contrast_n = 0.0;
  double composition = 0.0;
  double exposure_penalty = 0.0;
  double score = 0.0;
};

Aesthetic score_frame(const cv::Mat& frame, const Box* subject, double ref_sharpness = 200.0);

}  // namespace ed
