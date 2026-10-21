// 镜头切分后的处理：逐帧概率 -> 切点时间。
//
// 模型只给每帧一个 logit，真正的"哪一帧算切点"在这里做：阈值、局部极大、
// 300ms 抑制。逻辑独立于 ONNX Runtime，方便直接拿一串概率做单测。
#pragma once

#include <vector>

namespace ed {

inline std::vector<double> pick_shot_boundaries(const std::vector<float>& prob, double fps,
                                                float threshold, double min_gap_s = 0.3) {
  std::vector<double> boundaries;
  if (prob.size() < 3 || fps <= 0.0) return boundaries;
  const int n = static_cast<int>(prob.size());
  const int gap = static_cast<int>(fps * min_gap_s);

  for (int i = 1; i + 1 < n; ++i) {
    if (prob[i] < threshold) continue;
    if (prob[i] < prob[i - 1] || prob[i] < prob[i + 1]) continue;
    boundaries.push_back(i / fps);
    i += gap;
  }
  return boundaries;
}

}  // namespace ed
