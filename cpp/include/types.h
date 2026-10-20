// 自动剪辑用到的公共类型。
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace ed {

struct Box {
  float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
  float score = 0.f;
  int cls = 0;

  float area() const { return std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1); }
  float cx() const { return (x1 + x2) * 0.5f; }
  float cy() const { return (y1 + y2) * 0.5f; }
  const char* label() const;
};

struct VideoInfo {
  std::string path;
  int width = 0;
  int height = 0;
  double fps = 25.0;
  double duration = 0.0;
  long n_frames = 0;
};

struct Segment {
  double start = 0.0;
  double end = 0.0;
  double score = 0.0;
  double protagonist_ratio = 0.0;
  double aesthetic = 0.0;
  double duration() const { return std::max(0.0, end - start); }
};

const char* coco80(int idx);

}  // namespace ed
