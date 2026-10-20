// BoT-SORT：8 维 Kalman + IoU/外观代价 + 贪心匹配 + ORB 全局运动补偿。
#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "types.h"

namespace ed {

class KalmanBox {
 public:
  explicit KalmanBox(const Box& b);
  cv::Vec2f predict(const cv::Vec2f* shift);
  void update(const Box& b);
  cv::Rect2f xyxy() const;

 private:
  cv::Mat x_;  // 8x1
  cv::Mat P_;
  cv::Mat F_, H_, Q_, R_;
};

struct Track {
  int tid = 0;
  int cls = 0;
  KalmanBox kf;
  float score = 0.f;
  int hits = 1;
  int age = 0;
  int time_since_update = 0;
  bool confirmed = false;
  cv::Mat feature;  // 1xN，L2 归一化
  Track(const Box& b, int id) : tid(id), cls(b.cls), kf(b), score(b.score) {}
  Box as_box() const;
};

class BoTSORT {
 public:
  explicit BoTSORT(float iou_gate = 0.30f, float appearance_weight = 0.35f,
                   int max_age = 30, int min_hits = 2, bool use_gmc = true);

  // 输入当前帧检测与外观特征（可为空），返回已确认且在跟踪中的目标
  std::vector<Track> update(const std::vector<Box>& dets, const cv::Mat& feats,
                            const cv::Mat& gray);
  int next_id() const { return next_id_; }

 private:
  cv::Vec2f estimate_gmc(const cv::Mat& gray);
  void prune();

  float iou_gate_;
  float appearance_weight_;
  int max_age_;
  int min_hits_;
  bool use_gmc_;
  std::vector<Track> tracks_;
  int next_id_ = 1;
  cv::Mat prev_gray_;
  cv::Vec2f gmc_shift_{0.f, 0.f};
};

}  // namespace ed
