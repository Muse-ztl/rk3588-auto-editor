// YOLOv8 三个尺度的 DFL 解码。
//
// 单独拆出来是因为这块最容易出错：RKNN 输出的 layout 可能是 NCHW 也可能是
// NHWC，一旦索引写反不会报错，只会得到一堆乱框（或者一个框都没有）。
// 这里两种 layout 都显式支持，并且有 x86 单测覆盖。
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "types.h"

namespace ed {

// chw=true: [C][H][W]，false: [H][W][C]
inline float tensor_at(const float* p, int c, int y, int x, int H, int W, int C, bool chw) {
  if (chw) return p[(static_cast<size_t>(c) * H + y) * W + x];
  return p[(static_cast<size_t>(y) * W + x) * C + c];
}

// 一条边上 bins 个 logits 的 softmax 期望（0..bins-1），再乘 stride 得到像素距离。
inline float dfl_expect(const float* p, int off, int bins, bool chw, int y, int x, int H,
                        int W, int C) {
  float m = -1e9f;
  for (int b = 0; b < bins; ++b)
    m = std::max(m, tensor_at(p, off + b, y, x, H, W, C, chw));
  float sum = 0.f, acc = 0.f;
  for (int b = 0; b < bins; ++b) {
    const float e = std::exp(tensor_at(p, off + b, y, x, H, W, C, chw) - m);
    sum += e;
    acc += e * static_cast<float>(b);
  }
  return sum > 0.f ? acc / sum : 0.f;
}

// box: [4*bins][H][W] 或 [H][W][4*bins]；cls: [nc][H][W] 或 [H][W][nc]（已 sigmoid）。
// 输出坐标还是 letterbox 后的 640×640，由调用方再反算回原图。
inline void decode_dfl_scale(const float* box, const float* cls, int H, int W, int nc,
                             int bins, float stride, float conf, bool box_chw,
                             bool cls_chw, std::vector<Box>& out) {
  const int box_c = 4 * bins;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int best = -1;
      float best_p = conf;
      for (int c = 0; c < nc; ++c) {
        const float p = tensor_at(cls, c, y, x, H, W, nc, cls_chw);
        if (p > best_p) {
          best_p = p;
          best = c;
        }
      }
      if (best < 0) continue;

      const float cx = (static_cast<float>(x) + 0.5f) * stride;
      const float cy = (static_cast<float>(y) + 0.5f) * stride;
      Box b;
      b.x1 = cx - dfl_expect(box, 0 * bins, bins, box_chw, y, x, H, W, box_c) * stride;
      b.y1 = cy - dfl_expect(box, 1 * bins, bins, box_chw, y, x, H, W, box_c) * stride;
      b.x2 = cx + dfl_expect(box, 2 * bins, bins, box_chw, y, x, H, W, box_c) * stride;
      b.y2 = cy + dfl_expect(box, 3 * bins, bins, box_chw, y, x, H, W, box_c) * stride;
      b.score = best_p;
      b.cls = best;
      out.push_back(b);
    }
  }
}

}  // namespace ed
