#include "aesthetics.h"

#include <opencv2/imgproc.hpp>

#include "test_framework.h"

using namespace ed;

TEST(aesthetics_prefers_sharp_images) {
  cv::Mat sharp(360, 640, CV_8UC3);
  cv::randu(sharp, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
  cv::Mat blurred;
  cv::GaussianBlur(sharp, blurred, cv::Size(31, 31), 0);

  const Aesthetic a = score_frame(sharp, nullptr);
  const Aesthetic b = score_frame(blurred, nullptr);
  CHECK(a.sharpness > 2.0 * b.sharpness);
  CHECK(a.score > b.score);
}

TEST(aesthetics_penalizes_near_black_frames) {
  cv::Mat dark(180, 320, CV_8UC3, cv::Scalar(2, 2, 2));
  cv::Mat mid(180, 320, CV_8UC3, cv::Scalar(128, 128, 128));
  const Aesthetic d = score_frame(dark, nullptr);
  const Aesthetic m = score_frame(mid, nullptr);
  CHECK(d.exposure_penalty > 0.8);
  CHECK(m.exposure_penalty < 0.05);
  CHECK(d.score < m.score);
}

TEST(aesthetics_rewards_rule_of_thirds_subject) {
  cv::Mat img(360, 640, CV_8UC3, cv::Scalar(80, 120, 160));
  Box center;
  center.x1 = 290;
  center.y1 = 150;
  center.x2 = 350;
  center.y2 = 320;
  Box corner;
  corner.x1 = 10;
  corner.y1 = 10;
  corner.x2 = 70;
  corner.y2 = 80;
  const Aesthetic c = score_frame(img, &center);
  const Aesthetic e = score_frame(img, &corner);
  CHECK(c.composition > e.composition);
}

TEST(hsv_features_are_normalized_per_box) {
  cv::Mat img(120, 160, CV_8UC3, cv::Scalar(10, 200, 200));
  Box a;
  a.x1 = 10;
  a.y1 = 10;
  a.x2 = 80;
  a.y2 = 100;
  const cv::Mat f = hsv_features(img, {a});
  CHECK_EQ(f.rows, 1);
  CHECK_EQ(f.cols, 32);
  CHECK_NEAR(cv::norm(f.row(0)), 1.0, 1e-4);

  const cv::Mat empty = hsv_features(img, {});
  CHECK_EQ(empty.rows, 0);
}
