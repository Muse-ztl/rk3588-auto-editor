// 镜头切分：TransNetV2（100 帧窗口，27×48 输入），无模型时退化为直方图差分。
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "ort_model.h"

namespace ed {

class ShotDetector {
 public:
  // model_path 为空或加载失败时，detect() 会自动走直方图回退
  explicit ShotDetector(const std::string& model_path, float threshold = 0.5f);

  bool using_transnet() const { return model_ && model_->ready(); }

  // 返回镜头边界时间点（秒），不含 0 和片尾
  // shot_fps：预扫描采样帧率（默认 8fps 足够检测硬切，成本比逐帧低 3~4 倍）
  std::vector<double> detect(const std::string& video, double fps, int width, int height,
                             double shot_fps = 8.0);

 private:
  std::vector<double> detect_transnet(const std::vector<cv::Mat>& small_frames, double fps);
  std::vector<double> detect_histogram(const std::string& video);

  std::unique_ptr<OrtModel> model_;
  float threshold_;
};

}  // namespace ed
