#include "highlight.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ed {

double TrackStat::protagonist_score() const {
  // 曝光时长 + 中心度 + 清晰度 + 正脸 + 置信度（与 Python 版一致）
  return 0.34 * std::min(1.0, frames / 60.0) + 0.28 * mean_salience() +
         0.20 * mean_sharp() + 0.10 * std::min(1.0, face_hits / 10.0) +
         0.08 * mean_score();
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
  for (size_t i = 0; i < buckets.size(); ++i) raw.push_back((1.0 - aes_weight_) * ratios[i] + aes_weight_ * aes[i]);
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

  std::vector<Segment> cands;
  for (int i = 0; i < std::max(1, n - win + 1); ++i) {
    const int end = std::min(n, i + win);
    double sc = 0, pr = 0, ae = 0;
    const int cnt = end - i;
    for (int k = i; k < end; ++k) {
      sc += smooth[k];
      pr += ratios[k];
      ae += aes[k];
    }
    Segment s;
    s.start = buckets[i] * seg_seconds_;
    s.end = std::min(total_duration, s.start + win * seg_seconds_);
    s.score = sc / cnt;
    s.protagonist_ratio = pr / cnt;
    s.aesthetic = ae / cnt;
    cands.push_back(s);
  }
  if (cands.empty()) return {};

  const double target = std::max(min_clip_, total_duration * target_ratio_);
  const double budget = target * 1.15;
  std::vector<Segment> sorted = cands;
  std::sort(sorted.begin(), sorted.end(), [](const Segment& a, const Segment& b) {
    return a.score > b.score;
  });

  std::vector<Segment> chosen;
  for (const auto& s : sorted) {
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
  if (chosen.empty()) chosen.push_back(sorted.front());

  std::sort(chosen.begin(), chosen.end(), [](const Segment& a, const Segment& b) {
    return a.start < b.start;
  });
  return chosen;
}

}  // namespace ed
