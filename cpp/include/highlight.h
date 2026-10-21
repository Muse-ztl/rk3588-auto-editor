// 主角识别 + 片段打分 + EDL 生成。
#pragma once

#include <algorithm>
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
  int face_hits = 0;          // 命中正脸（frontal > 0.6）的次数
  double frontal_sum = 0.0;   // 每次命中的正脸程度累加
  double framing_sum = 0.0;   // 姿态构图分累加
  int framing_frames = 0;

  double mean_score() const { return score_sum / std::max(frames, 1); }
  double mean_salience() const { return salience_sum / std::max(frames, 1); }
  double mean_sharp() const { return sharp_sum / std::max(frames, 1); }
  double mean_frontal() const { return face_hits ? frontal_sum / face_hits : 0.0; }
  double mean_framing() const { return framing_frames ? framing_sum / framing_frames : 0.0; }
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
        // 三项权重必须加起来等于 1，否则"加分项"会变成"扣分项"
        aes_weight_(std::min(std::max(aes_weight, 0.0), 1.0 - face_weight_)) {}

  void add_frame(double t, const std::vector<Track>& tracks,
                 const std::map<int, double>& salience,
                 const std::map<int, double>& sharpness, double aesthetic, bool has_aes);

  // 人脸/姿态的逐 track 观测（在主循环里补充，打分时使用）
  // t 用于把正脸证据记到对应秒上：硬切/转身都会导致 track id 重建，
  // 按 id 记账会漏掉“换了 id 的同一个人”，按秒记账更稳。
  void add_face(double t, int tid, double frontal);
  void add_framing(int tid, double framing);

  void set_debug(bool on) { debug_ = on; }
  // 镜头边界（秒）：选段窗口不跨越硬切点，避免成片从镜头中间开始
  void set_boundaries(const std::vector<double>& b) { boundaries_ = b; }
  const TrackStat* choose_protagonist();
  std::vector<Segment> build_segments(double total_duration) const;

  int protagonist_tid() const { return protag_tid_; }
  const std::map<int, TrackStat>& stats() const { return stats_; }
  double aes_weight() const { return aes_weight_; }
  double face_weight() const { return face_weight_; }

 private:
  double fps_;
  double seg_seconds_;
  double min_clip_;
  double max_clip_;
  double target_ratio_;
  double face_weight_ = 0.16;
  double aes_weight_;

  std::map<int, TrackStat> stats_;
  std::map<int, int> bucket_frames_;
  std::map<int, double> bucket_prom_;   // 该秒内人物的最大显著性
  std::map<int, std::pair<double, int>> bucket_aes_;
  std::map<int, double> bucket_frontal_;
  int protag_tid_ = -1;
  bool debug_ = false;
  std::vector<double> boundaries_;
};

}  // namespace ed
