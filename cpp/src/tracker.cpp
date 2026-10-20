#include "tracker.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace ed {
namespace {

float iou_xyxy(const Box& a, const Box& b) {
  const float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
  return inter / (a.area() + b.area() - inter + 1e-9f);
}

float iou_rect(const cv::Rect2f& a, const Box& b) {
  const float x1 = std::max(a.x, b.x1), y1 = std::max(a.y, b.y1);
  const float x2 = std::min(a.x + a.width, b.x2), y2 = std::min(a.y + a.height, b.y2);
  const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
  const float aa = std::max(0.f, a.width) * std::max(0.f, a.height);
  return inter / (aa + b.area() - inter + 1e-9f);
}

}  // namespace

KalmanBox::KalmanBox(const Box& b) {
  F_ = cv::Mat::eye(8, 8, CV_32F);
  for (int i = 0; i < 4; ++i) F_.at<float>(i, i + 4) = 1.f;
  H_ = cv::Mat::zeros(4, 8, CV_32F);
  for (int i = 0; i < 4; ++i) H_.at<float>(i, i) = 1.f;
  Q_ = cv::Mat::eye(8, 8, CV_32F) * 0.01f;
  R_ = cv::Mat::eye(4, 4, CV_32F) * 0.1f;
  P_ = cv::Mat::eye(8, 8, CV_32F) * 10.f;
  for (int i = 4; i < 8; ++i) P_.at<float>(i, i) *= 1000.f;

  x_ = cv::Mat::zeros(8, 1, CV_32F);
  x_.at<float>(0) = (b.x1 + b.x2) * 0.5f;
  x_.at<float>(1) = (b.y1 + b.y2) * 0.5f;
  x_.at<float>(2) = b.x2 - b.x1;
  x_.at<float>(3) = b.y2 - b.y1;
}

cv::Vec2f KalmanBox::predict(const cv::Vec2f* shift) {
  if (shift != nullptr) {
    x_.at<float>(0) += shift->val[0];
    x_.at<float>(1) += shift->val[1];
  }
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.t() + Q_;
  return cv::Vec2f(x_.at<float>(0), x_.at<float>(1));
}

void KalmanBox::update(const Box& b) {
  cv::Mat z(4, 1, CV_32F);
  z.at<float>(0) = (b.x1 + b.x2) * 0.5f;
  z.at<float>(1) = (b.y1 + b.y2) * 0.5f;
  z.at<float>(2) = b.x2 - b.x1;
  z.at<float>(3) = b.y2 - b.y1;

  cv::Mat y = z - H_ * x_;
  cv::Mat S = H_ * P_ * H_.t() + R_;
  cv::Mat K = P_ * H_.t() * S.inv();
  x_ = x_ + K * y;
  P_ = (cv::Mat::eye(8, 8, CV_32F) - K * H_) * P_;
}

cv::Rect2f KalmanBox::xyxy() const {
  const float cx = x_.at<float>(0), cy = x_.at<float>(1);
  const float w = std::max(1.f, x_.at<float>(2)), h = std::max(1.f, x_.at<float>(3));
  return cv::Rect2f(cx - w / 2, cy - h / 2, w, h);
}

Box Track::as_box() const {
  const auto r = kf.xyxy();
  Box b;
  b.x1 = r.x;
  b.y1 = r.y;
  b.x2 = r.x + r.width;
  b.y2 = r.y + r.height;
  b.score = score;
  b.cls = cls;
  return b;
}

BoTSORT::BoTSORT(float iou_gate, float appearance_weight, int max_age, int min_hits,
                 bool use_gmc)
    : iou_gate_(iou_gate),
      appearance_weight_(appearance_weight),
      max_age_(max_age),
      min_hits_(min_hits),
      use_gmc_(use_gmc) {}

cv::Vec2f BoTSORT::estimate_gmc(const cv::Mat& gray) {
  if (prev_gray_.empty() || prev_gray_.size() != gray.size()) {
    prev_gray_ = gray.clone();
    return cv::Vec2f(0.f, 0.f);
  }
  try {
    auto orb = cv::ORB::create(200);
    std::vector<cv::KeyPoint> k1, k2;
    cv::Mat d1, d2;
    orb->detectAndCompute(prev_gray_, cv::noArray(), k1, d1);
    orb->detectAndCompute(gray, cv::noArray(), k2, d2);
    if (!d1.empty() && !d2.empty() && k1.size() > 10 && k2.size() > 10) {
      cv::BFMatcher bf(cv::NORM_HAMMING, true);
      std::vector<cv::DMatch> matches;
      bf.match(d1, d2, matches);
      if (matches.size() > 12) {
        std::vector<cv::Point2f> p1, p2;
        for (const auto& m : matches) {
          p1.push_back(k1[m.queryIdx].pt);
          p2.push_back(k2[m.trainIdx].pt);
        }
        cv::Mat affine;
        affine = cv::estimateAffinePartial2D(p1, p2, cv::noArray(), cv::RANSAC);
        if (!affine.empty()) gmc_shift_ = cv::Vec2f(affine.at<double>(0, 2),
                                                    affine.at<double>(1, 2));
      }
    }
  } catch (const cv::Exception&) {
    // GMC 只是补偿，失败就当没有位移
  }
  prev_gray_ = gray.clone();
  return gmc_shift_;
}

std::vector<Track> BoTSORT::update(const std::vector<Box>& dets, const cv::Mat& feats,
                                   const cv::Mat& gray) {
  cv::Vec2f shift(0.f, 0.f);
  const bool has_shift = use_gmc_ && !gray.empty();
  if (has_shift) shift = estimate_gmc(gray);

  for (auto& t : tracks_) {
    t.kf.predict(has_shift ? &shift : nullptr);
    ++t.age;
    ++t.time_since_update;
  }

  if (dets.empty()) {
    prune();
    std::vector<Track> out;
    for (auto& t : tracks_)
      if (t.confirmed && t.time_since_update == 0) out.push_back(t);
    return out;
  }

  const size_t nt = tracks_.size(), nd = dets.size();
  std::vector<std::vector<float>> cost(nt, std::vector<float>(nd, 1e5f));
  for (size_t i = 0; i < nt; ++i) {
    const auto rect = tracks_[i].kf.xyxy();
    for (size_t j = 0; j < nd; ++j) {
      const float iou = iou_rect(rect, dets[j]);
      if (iou < iou_gate_) continue;
      float c = 1.f - iou;
      if (!feats.empty() && !tracks_[i].feature.empty()) {
        const float sim = tracks_[i].feature.dot(feats.row(static_cast<int>(j)));
        c = (1.f - appearance_weight_) * c + appearance_weight_ * (1.f - sim);
      }
      cost[i][j] = c;
    }
  }

  // 贪心匹配：目标数 <20 时与匈牙利结果基本一致，少一份依赖
  std::vector<std::pair<float, std::pair<int, int>>> order;
  order.reserve(nt * nd);
  for (size_t i = 0; i < nt; ++i)
    for (size_t j = 0; j < nd; ++j)
      if (cost[i][j] < 1e4f) order.push_back({cost[i][j], {static_cast<int>(i), static_cast<int>(j)}});
  std::sort(order.begin(), order.end());

  std::vector<bool> trk_used(nt, false), det_used(nd, false);
  for (const auto& e : order) {
    const int i = e.second.first, j = e.second.second;
    if (trk_used[i] || det_used[j]) continue;
    trk_used[i] = det_used[j] = true;
    auto& t = tracks_[i];
    t.kf.update(dets[j]);
    t.score = dets[j].score;
    ++t.hits;
    t.time_since_update = 0;
    if (!feats.empty()) {
      cv::Mat f = feats.row(j).clone();
      if (t.feature.empty()) {
        t.feature = f;
      } else {
        t.feature = 0.9f * t.feature + 0.1f * f;
        t.feature /= cv::norm(t.feature) + 1e-9f;
      }
    }
    if (t.hits >= min_hits_) t.confirmed = true;
  }

  for (size_t j = 0; j < nd; ++j) {
    if (det_used[j] || dets[j].score < 0.25f) continue;
    Track t(dets[j], next_id_++);
    if (!feats.empty()) t.feature = feats.row(static_cast<int>(j)).clone();
    tracks_.push_back(t);
  }

  prune();
  std::vector<Track> out;
  for (auto& t : tracks_)
    if (t.confirmed && t.time_since_update == 0) out.push_back(t);
  return out;
}

void BoTSORT::prune() {
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [&](const Track& t) { return t.time_since_update > max_age_; }),
                tracks_.end());
}

}  // namespace ed
