// 延迟统计：报告里的 p50/p95 都用这里，避免各处各写一套。
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ed {

// 最近秩百分位：取排序后下标 floor(p/100 * (n-1))。
// 约定和旧版报告一致，历史数据还能对比；样本再少也不会越界。
inline double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double q = std::min(1.0, std::max(0.0, p / 100.0));
  const size_t i = static_cast<size_t>(q * static_cast<double>(v.size() - 1));
  return v[i];
}

inline double mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

}  // namespace ed
