#include "highlight.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ed {

double TrackStat::protagonist_score() const {
  // 接入人脸/姿态后重新配权：正脸与构图各占一档
  const double face_term = std::min(1.0, face_hits / 10.0) * (0.6 + 0.4 * mean_frontal());
  return 0.30 * std::min(1.0, frames / 60.0) + 0.24 * mean_salience() +
         0.16 * mean_sharp() + 0.12 * face_term + 0.10 * mean_framing() +
         0.08 * mean_score();
}

void HighlightPicker::add_face(double t, int tid, double frontal) {
  const int bucket = static_cast<int>(t / seg_seconds_);
  bucket_frontal_[bucket] = std::max(bucket_frontal_[bucket], frontal);

  auto it = stats_.find(tid);
  if (it == stats_.end()) return;
  if (frontal < 0.5) return;  // 只统计"看得出是正脸"的帧
  ++it->second.face_hits;
  it->second.frontal_sum += frontal;
}

void HighlightPicker::add_framing(int tid, double framing) {
  auto it = stats_.find(tid);
  if (it == stats_.end()) return;
  it->second.framing_sum += framing;
  ++it->second.framing_frames;
}

void HighlightPicker::add_frame(double t, const std::vector<Track>& tracks,
                                const std::map<int, double>& salience,
                                const std::map<int, double>& sharpness, double aesthetic,
                                bool has_aes) {
  const int bucket = static_cast<int>(t / seg_seconds_);
  ++bucket_frames_[bucket];

  double max_person_prom = 0.0;
  for (const auto& tr : tracks) {
    auto& st = stats_[tr.tid];
    st.tid = tr.tid;
    st.cls = tr.cls;
    ++st.frames;
    st.score_sum += tr.score;
    const double sal = salience.count(tr.tid) ? salience.at(tr.tid) : 0.0;
    st.salience_sum += sal;
    st.sharp_sum += sharpness.count(tr.tid) ? sharpness.at(tr.tid) : 0.0;
    if (tr.cls == 0) max_person_prom = std::max(max_person_prom, sal);
  }
  bucket_prom_[bucket] = std::max(bucket_prom_[bucket], max_person_prom);

  if (has_aes) {
    auto& v = bucket_aes_[bucket];
    v.first += aesthetic;
    v.second += 1;
  }
}

const TrackStat* HighlightPicker::choose_protagonist() {
  if (stats_.empty()) return nullptr;
  const TrackStat* best = nullptr;
  double best_score = -1.0;
  for (const auto& kv : stats_) {
    const auto& s = kv.second;
    if (s.cls != 0 || s.frames < 3) continue;
    const double sc = s.protagonist_score();
    if (sc > best_score) {
      best_score = sc;
      best = &s;
    }
  }
  if (best == nullptr) {
    for (const auto& kv : stats_) {
      const auto& s = kv.second;
      if (s.frames < 3) continue;
      const double sc = s.protagonist_score();
      if (sc > best_score) {
        best_score = sc;
        best = &s;
      }
    }
  }
  if (best != nullptr) protag_tid_ = best->tid;
  return best;
}

std::vector<Segment> HighlightPicker::build_segments(double total_duration) const {
  if (bucket_frames_.empty()) return {};

  std::vector<int> buckets;
  std::vector<double> aes, ratios;
  double last_aes = 0.0;
  for (const auto& kv : bucket_frames_) {
    buckets.push_back(kv.first);
    auto it = bucket_aes_.find(kv.first);
    if (it != bucket_aes_.end() && it->second.second > 0) {
      last_aes = it->second.first / it->second.second;
    }
    aes.push_back(last_aes);
    const double prom = bucket_prom_.count(kv.first) ? bucket_prom_.at(kv.first) : 0.0;
    ratios.push_back(std::min(1.0, prom / 0.75));
  }

  std::vector<double> raw, smooth;
  // 三项权重相加为 1：显著性为主，画质次之，正脸作为第三项证据（避免"没正脸=扣分"）
  const double base_w = 1.0 - aes_weight_ - face_weight_;
  for (size_t i = 0; i < buckets.size(); ++i) {
    const double face = bucket_frontal_.count(buckets[i]) ? bucket_frontal_.at(buckets[i]) : 0.0;
    raw.push_back(base_w * ratios[i] + aes_weight_ * aes[i] + face_weight_ * face);
  }
  for (size_t i = 0; i < raw.size(); ++i) {
    const size_t lo = i == 0 ? 0 : i - 1;
    const size_t hi = std::min(raw.size(), i + 2);
    double s = 0.0;
    for (size_t k = lo; k < hi; ++k) s += raw[k];
    smooth.push_back(s / static_cast<double>(hi - lo));
  }

  if (debug_) {
    printf("[buckets] %5s %6s %8s %6s %6s %7s\n", "sec", "frames", "prom", "ratio", "aes",
           "smooth");
    for (size_t i = 0; i < buckets.size(); ++i) {
      printf("[buckets] %5d %6d %8.3f %6.3f %6.3f %7.3f\n", buckets[i],
             bucket_frames_.at(buckets[i]),
             bucket_prom_.count(buckets[i]) ? bucket_prom_.at(buckets[i]) : 0.0, ratios[i],
             aes[i], smooth[i]);
    }
  }

  const double win_len = std::max(min_clip_, std::min(max_clip_, total_duration * target_ratio_));
  const int win = std::max(1, static_cast<int>(std::round(win_len / seg_seconds_)));
  const int n = static_cast<int>(buckets.size());

  // 候选窗口只在单个镜头内部滑：跨切点的话成片会从半个镜头开始
  std::vector<std::pair<double, double>> shots;
  {
    double prev = 0.0;
    for (double b : boundaries_) {
      if (b - prev > 1.0) shots.emplace_back(prev, b);
      prev = std::max(prev, b);
    }
    if (total_duration - prev > 1.0) shots.emplace_back(prev, total_duration);
    if (shots.empty()) shots.emplace_back(0.0, total_duration);
  }

  std::vector<Segment> cands;
  for (const auto& shot : shots) {
    std::vector<int> idxs;
    for (int i = 0; i < n; ++i) {
      const double t0 = buckets[i] * seg_seconds_;
      if (t0 >= shot.first - 1e-6 && t0 < shot.second - 1e-6) idxs.push_back(i);
    }
    if (idxs.size() < 2) continue;
    const int w = std::min(win, static_cast<int>(idxs.size()));
    for (int k = 0; k + w <= static_cast<int>(idxs.size()); ++k) {
      const int i0 = idxs[k], i1 = idxs[k + w - 1];
      double sc = 0, pr = 0, ae = 0;
      for (int j = k; j < k + w; ++j) {
        sc += smooth[idxs[j]];
        pr += ratios[idxs[j]];
        ae += aes[idxs[j]];
      }
      Segment seg;
      seg.start = buckets[i0] * seg_seconds_;
      seg.end = std::min(total_duration, (buckets[i1] + 1) * seg_seconds_);
      seg.score = sc / w;
      seg.protagonist_ratio = pr / w;
      seg.aesthetic = ae / w;
      cands.push_back(seg);
    }
  }
  if (cands.empty()) return {};

  const double target = std::max(min_clip_, total_duration * target_ratio_);
  const double budget = target * 1.15;
  std::vector<Segment> sorted = cands;
  std::sort(sorted.begin(), sorted.end(), [](const Segment& a, const Segment& b) {
    return a.score > b.score;
  });

  std::vector<Segment> chosen;
  auto crosses_boundary = [&](const Segment& s) {
    for (double b : boundaries_) {
      if (b > s.start + 0.15 && b < s.end - 0.15) return true;
    }
    return false;
  };

  // 两轮：先只选不跨镜头的窗口；一个都选不出来时才退让
  for (int pass = 0; pass < 2 && chosen.empty(); ++pass) {
    for (const auto& s : sorted) {
      if (pass == 0 && crosses_boundary(s)) continue;
      double used = 0.0;
      for (const auto& c : chosen) used += c.duration();
      if (used >= target) break;
      if (used + s.duration() > budget) continue;
      bool overlap = false;
      for (const auto& c : chosen) {
        if (s.start < c.end && c.start < s.end) {
          overlap = true;
          break;
        }
      }
      if (overlap) continue;
      chosen.push_back(s);
    }
  }
  if (chosen.empty()) chosen.push_back(sorted.front());

  std::sort(chosen.begin(), chosen.end(), [](const Segment& a, const Segment& b) {
    return a.start < b.start;
  });
  return chosen;
}

}  // namespace ed
