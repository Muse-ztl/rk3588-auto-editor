// 主角识别 + 片段打分 + EDL 生成。
#pragma once

#include <map>
#include <string>
#include <vector>

#include "tracker.h"
#include "types.h"

namespace ed {

struct TrackStat {
  int tid = 0;
  int cls = 0;
  int frames = 0;
  double score_sum = 0.0;
  double salience_sum = 0.0;
  double sharp_sum = 0.0;
  int face_hits = 0;

  double mean_score() const { return score_sum / std::max(frames, 1); }
  double mean_salience() const { return salience_sum / std::max(frames, 1); }
  double mean_sharp() const { return sharp_sum / std::max(frames, 1); }
  double protagonist_score() const;
};

class HighlightPicker {
 public:
  explicit HighlightPicker(double fps, double segment_seconds = 1.0, double min_clip = 3.0,
                           double max_clip = 8.0, double target_ratio = 0.45,
                           double aes_weight = 0.38)
      : fps_(fps),
        seg_seconds_(segment_seconds),
        min_clip_(min_clip),
        max_clip_(max_clip),
        target_ratio_(target_ratio),
        aes_weight_(aes_weight) {}

  void add_frame(double t, const std::vector<Track>& tracks,
                 const std::map<int, double>& salience,
                 const std::map<int, double>& sharpness, double aesthetic, bool has_aes);

  void set_debug(bool on) { debug_ = on; }
  const TrackStat* choose_protagonist();
  std::vector<Segment> build_segments(double total_duration) const;

  int protagonist_tid() const { return protag_tid_; }
  const std::map<int, TrackStat>& stats() const { return stats_; }

 private:
  double fps_;
  double seg_seconds_;
  double min_clip_;
  double max_clip_;
  double target_ratio_;
  double aes_weight_;

  std::map<int, TrackStat> stats_;
  std::map<int, int> bucket_frames_;
  std::map<int, double> bucket_prom_;   // 该秒内人物的最大显著性
  std::map<int, std::pair<double, int>> bucket_aes_;
  int protag_tid_ = -1;
  bool debug_ = false;
};

}  // namespace ed
