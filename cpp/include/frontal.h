// 人脸五点几何：判断侧脸程度。
//
// 输入顺序和 YuNet 一致：右眼、左眼、鼻尖、右嘴角、左嘴角（x,y 交替共 10 个数）。
// 板端和单测共用这份，避免"改了一处忘了另一处"。
#pragma once

#include <algorithm>
#include <cmath>

namespace ed {

inline float frontal_score(const float* pts) {
  const float eye_r_x = pts[0], eye_r_y = pts[1];
  const float eye_l_x = pts[2], eye_l_y = pts[3];
  const float nose_x = pts[4];
  const float mouth_r_x = pts[6];
  const float mouth_l_x = pts[8];

  const float eye_dx = std::abs(eye_l_x - eye_r_x);
  if (eye_dx < 1e-3f) return 0.f;
  const float eye_dy = std::abs(eye_l_y - eye_r_y) / eye_dx;
  const float nose_mid = std::abs(nose_x - (eye_l_x + eye_r_x) * 0.5f) / eye_dx;
  const float mouth_mid = std::abs((mouth_l_x + mouth_r_x) * 0.5f - nose_x) / eye_dx;
  const float symmetry = 1.f - std::min(1.f, nose_mid * 2.f);
  const float level = 1.f - std::min(1.f, eye_dy * 3.f);
  const float mouth_ok = 1.f - std::min(1.f, mouth_mid * 2.f);
  return std::clamp(0.45f * symmetry + 0.35f * level + 0.20f * mouth_ok, 0.f, 1.f);
}

}  // namespace ed
