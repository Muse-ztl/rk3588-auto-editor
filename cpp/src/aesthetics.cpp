#include "aesthetics.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace ed {
namespace {

double laplacian_var(const cv::Mat& gray) {
  cv::Mat lap;
  cv::Laplacian(gray, lap, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(lap, mean, stddev);
  return stddev.val[0] * stddev.val[0];
}

// Hasler-Süsstrunk colorfulness
double colorfulness(const cv::Mat& bgr) {
  cv::Mat f;
  bgr.convertTo(f, CV_32FC3);
  std::vector<cv::Mat> ch(3);
  cv::split(f, ch);
  cv::Mat rg = ch[2] - ch[1];
  cv::Mat yb = 0.5f * (ch[2] + ch[1]) - ch[0];
  cv::Scalar mrg, srg, myb, syb;
  cv::meanStdDev(rg, mrg, srg);
  cv::meanStdDev(yb, myb, syb);
  const double s = std::sqrt(srg.val[0] * srg.val[0] + syb.val[0] * syb.val[0]);
  const double m = std::sqrt(mrg.val[0] * mrg.val[0] + myb.val[0] * myb.val[0]);
  return s + 0.3 * m;
}

double exposure_penalty(const cv::Mat& gray) {
  int hist[256] = {0};
  for (int y = 0; y < gray.rows; ++y) {
    const uchar* row = gray.ptr<uchar>(y);
    for (int x = 0; x < gray.cols; ++x) ++hist[row[x]];
  }
  const double total = static_cast<double>(gray.rows) * gray.cols + 1e-9;
  double dark = 0, bright = 0;
  for (int i = 0; i < 12; ++i) dark += hist[i];
  for (int i = 244; i < 256; ++i) bright += hist[i];
  return std::min(1.0, (dark / total) * 2.0 + (bright / total) * 2.0);
}

double composition(const Box* subject, int w, int h) {
  if (subject == nullptr) return 0.35;
  const double nx = subject->cx() / std::max(w, 1);
  const double ny = subject->cy() / std::max(h, 1);
  const double tx[4] = {1.0 / 3, 2.0 / 3, 1.0 / 3, 2.0 / 3};
  const double ty[4] = {1.0 / 3, 1.0 / 3, 2.0 / 3, 2.0 / 3};
  double d = 1e9;
  for (int i = 0; i < 4; ++i) d = std::min(d, std::hypot(nx - tx[i], ny - ty[i]));
  const double rule = std::max(0.0, 1.0 - d / 0.6);

  const double area = subject->area() / std::max(1.0, static_cast<double>(w) * h);
  double size_score;
  if (area < 0.08) size_score = area / 0.08;
  else if (area > 0.45) size_score = std::max(0.0, 1.0 - (area - 0.45) / 0.4);
  else size_score = 1.0;

  return 0.55 * rule + 0.45 * size_score;
}

}  // namespace

cv::Mat hsv_features(const cv::Mat& frame, const std::vector<Box>& boxes) {
  cv::Mat feats(static_cast<int>(boxes.size()), 32, CV_32F);
  if (boxes.empty()) return feats;

  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  const int w = frame.cols, h = frame.rows;

  for (size_t i = 0; i < boxes.size(); ++i) {
    const int x1 = std::max(0, std::min(w - 1, static_cast<int>(boxes[i].x1)));
    const int y1 = std::max(0, std::min(h - 1, static_cast<int>(boxes[i].y1)));
    const int x2 = std::max(0, std::min(w, static_cast<int>(boxes[i].x2)));
    const int y2 = std::max(0, std::min(h, static_cast<int>(boxes[i].y2)));
    cv::Mat row = feats.row(static_cast<int>(i));
    if (x2 - x1 < 4 || y2 - y1 < 4) {
      row.setTo(0.f);
      continue;
    }
    cv::Mat crop = hsv(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    const int channels[3] = {0, 1, 2};
    const int bins[3] = {16, 8, 8};
    const float range_h[] = {0, 180};
    const float range_sv[] = {0, 256};
    const float* ranges[3] = {range_h, range_sv, range_sv};
    int offset = 0;
    for (int c = 0; c < 3; ++c) {
      float* dst = row.ptr<float>(0) + offset;
      cv::Mat hist(1, bins[c], CV_32F, dst);
      int ch = channels[c];
      cv::calcHist(&crop, 1, &ch, cv::noArray(), hist, 1, &bins[c],
                   &ranges[c]);
      offset += bins[c];
    }
    const float n = static_cast<float>(cv::norm(row)) + 1e-9f;
    row /= n;
  }
  return feats;
}

Aesthetic score_frame(const cv::Mat& frame, const Box* subject, double ref_sharpness) {
  cv::Mat small;
  cv::resize(frame, small, cv::Size(160, 90), 0, 0, cv::INTER_AREA);
  cv::Mat gray;
  cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);

  Aesthetic a;
  a.sharpness = laplacian_var(gray);
  a.colorfulness = colorfulness(small);
  a.exposure_penalty = exposure_penalty(gray);
  const cv::Scalar mean = cv::mean(gray);
  cv::Scalar m, s;
  cv::meanStdDev(gray, m, s);
  a.contrast_n = std::min(1.0, s.val[0] / 64.0);

  a.sharpness_n = std::min(1.0, a.sharpness / std::max(ref_sharpness, 1e-6));
  a.colorfulness_n = std::min(1.0, a.colorfulness / 60.0);
  a.composition = composition(subject, frame.cols, frame.rows);

  a.score = 0.32 * a.sharpness_n + 0.20 * a.colorfulness_n + 0.14 * a.contrast_n +
            0.20 * a.composition + 0.14 * (1.0 - a.exposure_penalty);
  a.score = std::max(0.0, std::min(1.0, a.score));
  (void)mean;
  return a;
}

}  // namespace ed
