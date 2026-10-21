#include "shot.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "shot_math.h"

namespace ed {
namespace {

constexpr int kW = 48;
constexpr int kH = 27;
constexpr int kWin = 100;
constexpr int kStride = 50;

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

}  // namespace

ShotDetector::ShotDetector(const std::string& model_path, float threshold)
    : threshold_(threshold) {
  if (!model_path.empty()) model_ = std::make_unique<OrtModel>(model_path, 2);
}

std::vector<double> ShotDetector::detect(const std::string& video, double fps, int width,
                                         int height, double shot_fps) {
  if (using_transnet()) {
    cv::VideoCapture cap(video);
    if (!cap.isOpened()) return {};
    const int step = std::max(1, static_cast<int>(std::round(fps / shot_fps)));
    const double eff_fps = fps / step;
    std::vector<cv::Mat> small;
    small.reserve(static_cast<size_t>(eff_fps * 60));
    cv::Mat frame, resized, rgb;
    long i = 0;
    while (cap.read(frame)) {
      if (i++ % step != 0) continue;
      cv::resize(frame, resized, cv::Size(kW, kH), 0, 0, cv::INTER_AREA);
      cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
      small.push_back(rgb.clone());
    }
    cap.release();
    (void)width;
    (void)height;
    return detect_transnet(small, eff_fps);
  }
  return detect_histogram(video);
}

std::vector<double> ShotDetector::detect_transnet(const std::vector<cv::Mat>& frames,
                                                  double fps) {
  std::vector<double> boundaries;
  if (frames.empty()) return boundaries;

  const int n = static_cast<int>(frames.size());
  std::vector<float> prob(n, 0.f);
  std::vector<float> weight(n, 0.f);

  for (int start = 0; start < n; start += kStride) {
    std::vector<uint8_t> buf(static_cast<size_t>(kWin) * kH * kW * 3);
    for (int i = 0; i < kWin; ++i) {
      const int idx = std::min(start + i, n - 1);  // 尾部用最后一帧补齐
      const cv::Mat& f = frames[idx];
      std::memcpy(buf.data() + static_cast<size_t>(i) * kH * kW * 3, f.data,
                  static_cast<size_t>(kH) * kW * 3);
    }
    std::vector<int64_t> shape;
    const auto out = model_->run_u8(buf, &shape);
    for (int i = 0; i < kWin; ++i) {
      const int idx = start + i;
      if (idx >= n) break;
      prob[idx] += sigmoid(out[i]);
      weight[idx] += 1.f;
    }
    if (start + kWin >= n) break;
  }

  for (int i = 0; i < n; ++i) {
    if (weight[i] > 0) prob[i] /= weight[i];
  }
  // 局部极大 + 阈值，避免一个切换点被检测成多帧
  return pick_shot_boundaries(prob, fps, threshold_);
}

std::vector<double> ShotDetector::detect_histogram(const std::string& video) {
  std::vector<double> boundaries;
  cv::VideoCapture cap(video);
  if (!cap.isOpened()) return boundaries;
  const double fps = cap.get(cv::CAP_PROP_FPS);
  std::vector<double> dist;
  cv::Mat prev, frame, small;
  std::vector<cv::Mat> hists;

  while (cap.read(frame)) {
    cv::resize(frame, small, cv::Size(64, 36), 0, 0, cv::INTER_AREA);
    cv::Mat hsv;
    cv::cvtColor(small, hsv, cv::COLOR_BGR2HSV);
    int ch[2] = {0, 1};
    int bins[2] = {32, 16};
    float r0[] = {0, 180};
    float r1[] = {0, 256};
    const float* ranges[2] = {r0, r1};
    cv::Mat hist;
    cv::calcHist(&hsv, 1, ch, cv::noArray(), hist, 2, bins, ranges);
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    if (!prev.empty()) dist.push_back(cv::compareHist(prev, hist, cv::HISTCMP_BHATTACHARYYA));
    prev = hist.clone();
  }
  cap.release();
  if (dist.size() < 10) return boundaries;

  double mean = 0, var = 0;
  for (double d : dist) mean += d;
  mean /= dist.size();
  for (double d : dist) var += (d - mean) * (d - mean);
  const double thr = mean + 3.0 * std::sqrt(var / dist.size());
  for (size_t i = 1; i + 1 < dist.size(); ++i) {
    if (dist[i] > thr && dist[i] >= dist[i - 1] && dist[i] >= dist[i + 1]) {
      boundaries.push_back((i + 1) / fps);
    }
  }
  return boundaries;
}

}  // namespace ed
