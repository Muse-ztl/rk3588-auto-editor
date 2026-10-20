// 端侧视频自动剪辑（C++）：分析 pass + 出片 pass。
//
//   ./auto_editor --input data/demo_3scenes.mp4 --output output/final_cut.mp4 \
//                 --model models/yolov8n.rknn --analysis-fps 6 --report output/report.json
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "aesthetics.h"
#include "detector.h"
#include "highlight.h"
#include "tracker.h"
#include "types.h"
#include "video_io.h"

namespace {

struct Args {
  std::string input = "data/demo_3scenes.mp4";
  std::string output = "output/final_cut.mp4";
  std::string model = "models/yolov8n.rknn";
  std::string report;
  float conf = 0.30f;
  double analysis_fps = 6.0;
  int core_mask = 0;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (k == "--input") a.input = next();
    else if (k == "--output") a.output = next();
    else if (k == "--model") a.model = next();
    else if (k == "--report") a.report = next();
    else if (k == "--conf") a.conf = std::atof(next().c_str());
    else if (k == "--analysis-fps") a.analysis_fps = std::atof(next().c_str());
    else if (k == "--core-mask") a.core_mask = std::atoi(next().c_str());
    else if (k == "-h" || k == "--help") {
      printf(
          "usage: auto_editor --input VID --output MP4 [--model yolov8n.rknn]\n"
          "                   [--analysis-fps 6] [--conf 0.3] [--report out.json]\n");
      std::exit(0);
    }
  }
  return a;
}

double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t i = static_cast<size_t>(std::min(1.0, std::max(0.0, p / 100.0)) * (v.size() - 1));
  return v[i];
}

double laplacian_var(const cv::Mat& gray) {
  cv::Mat lap;
  cv::Laplacian(gray, lap, CV_64F);
  cv::Scalar m, s;
  cv::meanStdDev(lap, m, s);
  return s.val[0] * s.val[0];
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  std::filesystem::create_directories(
      std::filesystem::path(a.output).parent_path().string());

  const ed::VideoInfo info = ed::probe(a.input);
  if (info.width == 0) {
    fprintf(stderr, "cannot open video: %s\n", a.input.c_str());
    return 1;
  }
  printf("[editor] %s  %.1fs  %dx%d @%.1ffps\n", a.input.c_str(), info.duration,
         info.width, info.height, info.fps);

  const double stride_d = std::max(1.0, std::round(info.fps / std::max(1.0, a.analysis_fps)));
  const int stride = static_cast<int>(stride_d);
  const double analysis_rate = info.fps / stride;

  ed::YoloV8 det(a.model, a.conf, 0.45f, 640, a.core_mask);
  ed::BoTSORT tracker;
  ed::HighlightPicker picker(analysis_rate);

  std::vector<double> detect_ms;
  long analyzed = 0, detect_calls = 0, reid_calls = 0, aes_calls = 0;

  const double t_start = static_cast<double>(cv::getTickCount());
  cv::VideoCapture cap(a.input);
  cv::Mat frame;
  long idx = 0;
  while (cap.read(frame)) {
    if (idx % stride != 0) {
      ++idx;
      continue;
    }
    const double t = idx / info.fps;
    auto boxes = det.detect(frame);
    detect_ms.push_back(det.last_ms());
    ++detect_calls;

    cv::Mat feats;
    if (idx % (stride * 2) == 0) {
      feats = ed::hsv_features(frame, boxes);
      ++reid_calls;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    auto tracks = tracker.update(boxes, feats, gray);

    std::map<int, double> salience, sharpness;
    const int w = frame.cols, h = frame.rows;
    for (const auto& tr : tracks) {
      const auto r = tr.kf.xyxy();
      const double cx = r.x + r.width / 2, cy = r.y + r.height / 2;
      const double centrality =
          std::max(0.0, 1.0 - std::hypot(cx - w / 2.0, cy - h / 2.0) /
                              std::hypot(w / 2.0, h / 2.0));
      const double area_ratio = (r.width * r.height) / std::max(1.0, static_cast<double>(w) * h);
      salience[tr.tid] =
          0.35 * centrality + 0.65 * std::sqrt(std::min(1.0, area_ratio / 0.40));

      const int x1 = std::max(0, static_cast<int>(r.x));
      const int y1 = std::max(0, static_cast<int>(r.y));
      const int x2 = std::min(w, static_cast<int>(r.x + r.width));
      const int y2 = std::min(h, static_cast<int>(r.y + r.height));
      if (x2 > x1 && y2 > y1) {
        cv::Mat crop = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        cv::Mat cgray;
        cv::cvtColor(crop, cgray, cv::COLOR_BGR2GRAY);
        sharpness[tr.tid] = std::min(1.0, laplacian_var(cgray) / 300.0);
      }
    }

    double aes = 0.0;
    bool has_aes = false;
    if (idx % (stride * 12) == 0) {
      const ed::Box* subject = nullptr;
      ed::Box best;
      for (const auto& tr : tracks) {
        if (tr.cls != 0) continue;
        if (subject == nullptr || tr.score > best.score) {
          best = tr.as_box();
          subject = &best;
        }
      }
      aes = ed::score_frame(frame, subject).score;
      has_aes = true;
      ++aes_calls;
    }

    picker.add_frame(t, tracks, salience, sharpness, aes, has_aes);
    ++analyzed;
    if (analyzed % 60 == 0) {
      printf("[analyze] %.1fs / %.1fs  tracks=%zu\n", t, info.duration, tracks.size());
    }
    ++idx;
  }
  cap.release();

  const double pass1 =
      (static_cast<double>(cv::getTickCount()) - t_start) / cv::getTickFrequency();
  const auto* protag = picker.choose_protagonist();
  auto segments = picker.build_segments(info.duration);
  if (segments.empty()) {
    fprintf(stderr, "no segment selected\n");
    return 1;
  }

  // ---- 出片 pass
  const double t_render = static_cast<double>(cv::getTickCount());
  const std::string enc = ed::pick_encoder();
  std::vector<std::string> parts;
  std::string err;
  for (size_t i = 0; i < segments.size(); ++i) {
    std::ostringstream part;
    part << a.output << ".seg" << std::setw(2) << std::setfill('0') << i << ".mp4";
    if (!ed::cut_segment(a.input, segments[i].start, segments[i].duration(), part.str(), enc, err)) {
      fprintf(stderr, "cut failed: %s\n", err.c_str());
      return 1;
    }
    parts.push_back(part.str());
  }
  if (!ed::concat_segments(parts, a.output, err)) {
    fprintf(stderr, "concat failed: %s\n", err.c_str());
    return 1;
  }
  for (const auto& p : parts) std::remove(p.c_str());
  const double pass2 =
      (static_cast<double>(cv::getTickCount()) - t_render) / cv::getTickFrequency();

  const auto out_info = ed::probe(a.output);
  const double total = pass1 + pass2;

  printf("\n=== edit report ===\n");
  printf("video          : %.1fs %dx%d @%.1ffps\n", info.duration, info.width, info.height, info.fps);
  printf("encode         : %s\n", enc.c_str());
  printf("detector       : rknn-npu  p50=%.1fms p90=%.1fms\n", pct(detect_ms, 50),
         pct(detect_ms, 90));
  printf("analysis       : %ld frames @ %.1ffps, %.1fs (detect %ld, reid %ld, aesthetic %ld)\n",
         analyzed, analysis_rate, pass1, detect_calls, reid_calls, aes_calls);
  if (protag) {
    printf("protagonist    : tid=%d frames=%d score=%.3f salience=%.3f\n", protag->tid,
           protag->frames, protag->protagonist_score(), protag->mean_salience());
  }
  printf("segments       : ");
  for (const auto& s : segments) printf("(%.1f-%.1f %.2f) ", s.start, s.end, s.score);
  printf("\noutput         : %.1fs in %.1fs\n", out_info.duration, pass2);
  printf("total          : %.1fs (%.2fx realtime)\n", total, info.duration / std::max(total, 1e-6));

  if (!a.report.empty()) {
    std::ofstream rep(a.report);
    rep << std::fixed << std::setprecision(3) << "{\n"
        << "  \"input\": \"" << a.input << "\",\n"
        << "  \"output\": \"" << a.output << "\",\n"
        << "  \"video\": {\"width\": " << info.width << ", \"height\": " << info.height
        << ", \"fps\": " << info.fps << ", \"duration\": " << info.duration << "},\n"
        << "  \"encode_backend\": \"" << enc << "\",\n"
        << "  \"analysis_fps\": " << analysis_rate << ",\n"
        << "  \"analyzed_frames\": " << analyzed << ",\n"
        << "  \"detect_calls\": " << detect_calls << ",\n"
        << "  \"reid_calls\": " << reid_calls << ",\n"
        << "  \"aesthetic_calls\": " << aes_calls << ",\n"
        << "  \"detect_ms_p50\": " << pct(detect_ms, 50) << ",\n"
        << "  \"detect_ms_p90\": " << pct(detect_ms, 90) << ",\n"
        << "  \"pass1_seconds\": " << pass1 << ",\n"
        << "  \"pass2_seconds\": " << pass2 << ",\n"
        << "  \"output_duration\": " << out_info.duration << ",\n"
        << "  \"speedup_vs_realtime\": " << info.duration / std::max(total, 1e-6) << ",\n";
    if (protag) {
      rep << "  \"protagonist\": {\"track_id\": " << protag->tid << ", \"frames\": "
          << protag->frames << ", \"score\": " << protag->protagonist_score() << "},\n";
    }
    rep << "  \"segments\": [";
    for (size_t i = 0; i < segments.size(); ++i) {
      rep << (i ? ", " : "") << "{\"start\": " << segments[i].start
          << ", \"end\": " << segments[i].end << ", \"score\": " << segments[i].score
          << ", \"protagonist_ratio\": " << segments[i].protagonist_ratio << "}";
    }
    rep << "]\n}\n";
  }
  return 0;
}
