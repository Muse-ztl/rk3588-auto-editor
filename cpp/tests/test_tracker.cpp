#include "tracker.h"

#include <vector>

#include "test_framework.h"

using namespace ed;

namespace {

Box box_at(float cx, float cy, float w = 40.f, float h = 100.f, float score = 0.9f) {
  Box b;
  b.x1 = cx - w / 2;
  b.y1 = cy - h / 2;
  b.x2 = cx + w / 2;
  b.y2 = cy + h / 2;
  b.score = score;
  b.cls = 0;
  return b;
}

}  // namespace

TEST(kalman_follows_a_constant_velocity_box) {
  KalmanBox kf(box_at(100.f, 200.f));
  for (int i = 1; i <= 8; ++i) {
    kf.predict(nullptr);
    kf.update(box_at(100.f + 10.f * i, 200.f));
  }
  const auto r = kf.xyxy();
  CHECK_NEAR(r.x + r.width / 2, 180.0, 8.0);
  CHECK_NEAR(r.width, 40.0, 8.0);
}

TEST(botsort_keeps_one_id_for_a_moving_person) {
  BoTSORT tracker(0.30f, 0.35f, 30, 2, false);
  int id = -1;
  int seen = 0;
  for (int i = 0; i < 15; ++i) {
    const auto out = tracker.update({box_at(100.f + 3.f * i, 200.f)}, cv::Mat(), cv::Mat());
    if (out.empty()) continue;
    ++seen;
    if (id < 0) id = out[0].tid;
    CHECK_EQ(out[0].tid, id);
    CHECK_EQ(out.size(), 1u);
  }
  CHECK(id > 0);
  CHECK(seen >= 13);
}

TEST(botsort_keeps_two_people_apart) {
  BoTSORT tracker(0.30f, 0.35f, 30, 2, false);
  int left_id = -1, right_id = -1;
  for (int i = 0; i < 10; ++i) {
    const auto out = tracker.update({box_at(100.f, 200.f), box_at(500.f, 200.f)},
                                    cv::Mat(), cv::Mat());
    if (out.size() != 2) continue;
    if (left_id < 0) {
      left_id = out[0].tid;
      right_id = out[1].tid;
    }
    CHECK(left_id != right_id);
  }
  CHECK(left_id > 0);
  CHECK(right_id > 0);
}

TEST(botsort_drops_a_lost_track_and_assigns_a_new_id) {
  BoTSORT tracker(0.30f, 0.35f, 5, 2, false);
  for (int i = 0; i < 4; ++i) tracker.update({box_at(200.f, 200.f)}, cv::Mat(), cv::Mat());

  for (int i = 0; i < 12; ++i) {
    const auto out = tracker.update({}, cv::Mat(), cv::Mat());
    CHECK(out.empty());
  }

  // 旧 ID 已被 prune，新目标从头计数
  tracker.update({box_at(200.f, 200.f)}, cv::Mat(), cv::Mat());
  const auto out = tracker.update({box_at(200.f, 200.f)}, cv::Mat(), cv::Mat());
  CHECK_EQ(out.size(), 1u);
  CHECK(out[0].tid > 1);
}

TEST(track_round_trips_its_kalman_box) {
  Track t(box_at(300.f, 250.f, 60.f, 120.f), 7);
  const Box b = t.as_box();
  CHECK_EQ(b.cls, 0);
  CHECK_NEAR(b.x1, 270.0, 1e-4);
  CHECK_NEAR(b.y1, 190.0, 1e-4);
  CHECK_NEAR(b.x2, 330.0, 1e-4);
  CHECK_NEAR(b.y2, 310.0, 1e-4);
}
