// DFL 解码 / 镜头切点 / 正脸几何：这三块都是纯计算，最容易写反且没有编译期保护。
#include "dfl.h"
#include "frontal.h"
#include "shot_math.h"
#include "test_framework.h"

using namespace ed;

TEST(dfl_decodes_chw_and_nhwc_the_same) {
  const int H = 1, W = 1, nc = 2, bins = 2;
  // 四条边都用 [0, 0]，softmax 期望 = 0.5，stride=8 时距离 = 4px
  std::vector<float> box_chw(4 * bins, 0.f);
  std::vector<float> cls_chw{0.20f, 0.90f};

  std::vector<float> box_nhwc(4 * bins, 0.f);
  std::vector<float> cls_nhwc{0.20f, 0.90f};

  std::vector<Box> a, b;
  decode_dfl_scale(box_chw.data(), cls_chw.data(), H, W, nc, bins, 8.f, 0.30f, true, true, a);
  decode_dfl_scale(box_nhwc.data(), cls_nhwc.data(), H, W, nc, bins, 8.f, 0.30f, false,
                   false, b);

  CHECK_EQ(a.size(), 1u);
  CHECK_EQ(b.size(), 1u);
  CHECK_EQ(a[0].cls, 1);
  CHECK_NEAR(a[0].score, 0.90, 1e-6);
  CHECK_NEAR(a[0].x1, 0.0, 1e-5);
  CHECK_NEAR(a[0].y1, 0.0, 1e-5);
  CHECK_NEAR(a[0].x2, 8.0, 1e-5);
  CHECK_NEAR(a[0].y2, 8.0, 1e-5);
  CHECK_NEAR(a[0].x1, b[0].x1, 1e-5);
  CHECK_NEAR(a[0].x2, b[0].x2, 1e-5);
}

TEST(dfl_uses_softmax_expectation_not_argmax) {
  const int H = 1, W = 1, nc = 1, bins = 2;
  // logits [0, 1] -> 期望 = 1/(1+e^-1) ≈ 0.731，乘 stride 3 = 2.19px
  std::vector<float> box{0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f};
  std::vector<float> cls{0.8f};
  std::vector<Box> out;
  decode_dfl_scale(box.data(), cls.data(), H, W, nc, bins, 3.f, 0.3f, true, true, out);
  CHECK_EQ(out.size(), 1u);
  const double expect = 1.0 / (1.0 + std::exp(-1.0)) * 3.0;
  CHECK_NEAR(out[0].x2, 1.5 + expect, 1e-4);
}

TEST(dfl_respects_confidence_threshold) {
  const int H = 1, W = 1, nc = 2, bins = 2;
  std::vector<float> box(4 * bins, 0.f);
  std::vector<float> cls{0.20f, 0.10f};
  std::vector<Box> out;
  decode_dfl_scale(box.data(), cls.data(), H, W, nc, bins, 8.f, 0.30f, true, true, out);
  CHECK(out.empty());
}

TEST(dfl_decodes_every_anchor_in_the_grid) {
  const int H = 2, W = 3, nc = 1, bins = 2;
  std::vector<float> box(4 * bins * H * W, 0.f);
  std::vector<float> cls(H * W, 0.9f);
  std::vector<Box> out;
  decode_dfl_scale(box.data(), cls.data(), H, W, nc, bins, 4.f, 0.3f, true, true, out);
  CHECK_EQ(out.size(), 6u);
  CHECK_NEAR(out[0].x1, 0.0, 1e-5);     // 第 0 个 anchor 中心 (2,2)，距离 2
  CHECK_NEAR(out[0].x2, 4.0, 1e-5);
  CHECK_NEAR(out[5].x1, 8.0, 1e-5);     // 第 5 个 anchor 中心 (10,6)
  CHECK_NEAR(out[5].y2, 8.0, 1e-5);
}

TEST(shot_boundaries_need_threshold_and_local_max) {
  std::vector<float> prob(80, 0.f);
  prob[16] = 0.90f;  // 2.0s
  prob[40] = 0.90f;  // 5.0s
  prob[41] = 0.80f;  // 紧跟其后，应被 300ms 抑制
  prob[60] = 0.20f;  // 低于阈值，忽略
  const auto b = pick_shot_boundaries(prob, 8.0, 0.5f);
  CHECK_EQ(b.size(), 2u);
  CHECK_NEAR(b[0], 2.0, 1e-9);
  CHECK_NEAR(b[1], 5.0, 1e-9);
}

TEST(shot_boundaries_ignore_short_input) {
  CHECK(pick_shot_boundaries({}, 8.0, 0.5f).empty());
  CHECK(pick_shot_boundaries({1.f, 1.f}, 8.0, 0.5f).empty());
}

TEST(frontal_is_high_for_symmetric_face) {
  // 右眼(40,50) 左眼(60,50) 鼻(50,58) 右嘴角(43,70) 左嘴角(57,70)
  const float pts[10] = {40, 50, 60, 50, 50, 58, 43, 70, 57, 70};
  CHECK_NEAR(frontal_score(pts), 1.0, 1e-5);
}

TEST(frontal_drops_for_profile_face) {
  const float front[10] = {40, 50, 60, 50, 50, 58, 43, 70, 57, 70};
  const float side[10] = {40, 50, 60, 48, 66, 56, 63, 70, 69, 70};
  CHECK(frontal_score(side) < frontal_score(front) - 0.3f);
  CHECK(frontal_score(side) < 0.6f);
}

TEST(frontal_handles_degenerate_eyes) {
  const float pts[10] = {50, 50, 50, 50, 50, 58, 43, 70, 57, 70};
  CHECK_NEAR(frontal_score(pts), 0.0, 1e-6);
}
