#include "highlight.h"

#include <map>
#include <vector>

#include "test_framework.h"

using namespace ed;

namespace {

Track person(int tid, float x1 = 400.f, float y1 = 200.f, float x2 = 600.f,
             float y2 = 700.f) {
  Box b;
  b.x1 = x1;
  b.y1 = y1;
  b.x2 = x2;
  b.y2 = y2;
  b.score = 0.9f;
  b.cls = 0;
  return Track(b, tid);
}

void add_bucket(HighlightPicker& p, int sec, double salience, double aes = 0.5) {
  const std::vector<Track> tracks{person(1)};
  p.add_frame(sec, tracks, {{1, salience}}, {{1, 0.6}}, aes, true);
}

}  // namespace

TEST(highlight_picks_the_more_salient_person) {
  HighlightPicker p(1.0);
  for (int t = 0; t < 60; ++t) {
    std::vector<Track> tracks{person(1)};
    if (t % 10 == 0) tracks.push_back(person(2, 10, 10, 60, 120));
    p.add_frame(t, tracks, {{1, 0.85}, {2, 0.30}}, {{1, 0.7}, {2, 0.3}}, 0.5, true);
  }
  const TrackStat* best = p.choose_protagonist();
  CHECK(best != nullptr);
  CHECK_EQ(p.protagonist_tid(), 1);
}

TEST(highlight_does_not_cross_a_cut) {
  HighlightPicker p(1.0);
  p.set_boundaries({4.0, 8.0});
  for (int t = 0; t < 12; ++t) add_bucket(p, t, t >= 8 ? 0.9 : 0.1);

  const auto segs = p.build_segments(12.0);
  CHECK_EQ(segs.size(), 1u);
  CHECK(segs[0].start >= 8.0 - 1e-6);
  CHECK(segs[0].end <= 12.0 + 1e-6);
}

TEST(highlight_treats_frontal_face_as_bonus) {
  auto score_with = [](bool with_face) {
    HighlightPicker p(1.0);
    for (int t = 0; t < 6; ++t) {
      add_bucket(p, t, 0.4, 0.5);
      if (with_face) p.add_face(t, 1, 1.0f);
    }
    const auto segs = p.build_segments(6.0);
    CHECK_EQ(segs.size(), 1u);
    return segs[0].score;
  };
  CHECK(score_with(true) > score_with(false) + 0.02);
}

TEST(highlight_clamps_extreme_aesthetic_weight) {
  HighlightPicker p(1.0, 1.0, 3.0, 8.0, 0.45, 1.8);
  CHECK_NEAR(p.aes_weight(), 0.84, 1e-9);
  CHECK(p.aes_weight() + p.face_weight() <= 1.0 + 1e-9);

  for (int t = 0; t < 8; ++t) add_bucket(p, t, 0.0, 1.0);
  const auto segs = p.build_segments(8.0);
  CHECK_EQ(segs.size(), 1u);
  CHECK(segs[0].score >= 0.0);
  CHECK(segs[0].score <= 1.0);
}

TEST(highlight_returns_no_segment_without_frames) {
  HighlightPicker p(1.0);
  CHECK(p.build_segments(10.0).empty());
}

TEST(highlight_segments_do_not_overlap_or_exceed_budget) {
  HighlightPicker p(1.0);
  for (int t = 0; t < 30; ++t) add_bucket(p, t, 0.6, 0.5);
  const auto segs = p.build_segments(30.0);
  CHECK(!segs.empty());

  double total = 0.0;
  for (size_t i = 0; i < segs.size(); ++i) {
    total += segs[i].duration();
    CHECK(segs[i].end > segs[i].start);
    if (i) CHECK(segs[i].start >= segs[i - 1].end - 1e-6);
  }
  // 目标时长 13.5s，允许 15% 余量
  CHECK(total <= 13.5 * 1.15 + 1e-6);
}
