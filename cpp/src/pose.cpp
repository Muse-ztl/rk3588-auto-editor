#include "pose.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace ed {
namespace {

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
  return inter / (a.width * a.height + b.width * b.height - inter + 1e-9f);
}

}  // namespace

PoseEstimator::PoseEstimator(const std::string& model_path, int threads) {
  if (!model_path.empty()) model_ = std::make_unique<OrtModel>(model_path, threads);
}

PoseResult PoseEstimator::estimate(const cv::Mat& frame, const cv::Rect2f& person_box) {
  PoseResult r;
  if (!ready() || frame.empty()) return r;

  // letterbox 到 640×640
  const float ratio = std::min(static_cast<float>(imgsz_) / frame.rows,
                               static_cast<float>(imgsz_) / frame.cols);
  const int nh = static_cast<int>(std::round(frame.rows * ratio));
  const int nw = static_cast<int>(std::round(frame.cols * ratio));
  const float pad_x = (imgsz_ - nw) / 2.f;
  const float pad_y = (imgsz_ - nh) / 2.f;

  cv::Mat resized, canvas(imgsz_, imgsz_, CV_8UC3, cv::Scalar(114, 114, 114)), rgb;
  cv::resize(frame, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
  resized.copyTo(canvas(cv::Rect(static_cast<int>(pad_x), static_cast<int>(pad_y), nw, nh)));
  cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  std::vector<float> input(static_cast<size_t>(3) * imgsz_ * imgsz_);
  for (int y = 0; y < imgsz_; ++y) {
    const cv::Vec3f* row = rgb.ptr<cv::Vec3f>(y);
    for (int x = 0; x < imgsz_; ++x) {
      for (int c = 0; c < 3; ++c) {
        input[(static_cast<size_t>(c) * imgsz_ + y) * imgsz_ + x] = row[x][c];
      }
    }
  }

  std::vector<int64_t> shape;
  const auto out = model_->run(input, &shape);
  if (shape.size() != 3 || out.empty()) return r;

  const int channels = static_cast<int>(shape[1]);   // 56
  const int anchors = static_cast<int>(shape[2]);    // 8400
  if (channels < 56) return r;

  int best = -1;
  float best_iou = 0.f;
  for (int j = 0; j < anchors; ++j) {
    const float score = out[static_cast<size_t>(4) * anchors + j];
    if (score < 0.35f) continue;
    const float cx = out[0 * anchors + j], cy = out[1 * anchors + j];
    const float w = out[2 * anchors + j], h = out[3 * anchors + j];
    cv::Rect2f box((cx - w / 2 - pad_x) / ratio, (cy - h / 2 - pad_y) / ratio, w / ratio,
                   h / ratio);
    const float v = iou(box, person_box);
    if (v > best_iou) {
      best_iou = v;
      best = j;
    }
  }
  if (best < 0 || best_iou < 0.20f) return r;

  r.ok = true;
  r.score = out[static_cast<size_t>(4) * anchors + best];
  int visible = 0;
  for (int k = 0; k < 17; ++k) {
    const float kx = out[static_cast<size_t>(5 + k * 3) * anchors + best];
    const float ky = out[static_cast<size_t>(6 + k * 3) * anchors + best];
    const float kc = out[static_cast<size_t>(7 + k * 3) * anchors + best];
    r.kpts[k] = cv::Point2f((kx - pad_x) / ratio, (ky - pad_y) / ratio);
    r.conf[k] = kc;
    if (kc > 0.3f && r.kpts[k].x > 0 && r.kpts[k].x < frame.cols && r.kpts[k].y > 0 &&
        r.kpts[k].y < frame.rows) {
      ++visible;
    }
  }

  // 构图分：头部（鼻子）落在画面上三分之一处最舒服；关键点完整度越高越好
  const float head_y = r.kpts[0].y / std::max(1, frame.rows);
  const float head_score = std::max(0.f, 1.f - std::abs(head_y - 0.33f) / 0.33f);
  const float complete = visible / 17.f;
  r.framing = std::clamp(0.5f * head_score + 0.5f * complete, 0.f, 1.f);
  return r;
}

}  // namespace ed
