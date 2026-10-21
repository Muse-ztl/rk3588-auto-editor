// 端侧视频自动剪辑（C++）：分析 pass + 出片 pass。
//
//   ./auto_editor --input data/demo_3scenes.mp4 --output output/final_cut.mp4 \
//                 --model models/yolov8n.rknn --face-model models/yunet.onnx \
//                 --pose-model models/yolov8n-pose.onnx --shot-model models/transnetv2.onnx
//
// 模型都是可选的：缺哪个就跳过哪个模块（切分退化为直方图），主流程不受影响。
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
#include "face.h"
#include "highlight.h"
#include "pose.h"
#include "shot.h"
#include "stats.h"
#include "sysinfo.h"
#include "tracker.h"
#include "types.h"
#include "video_io.h"

namespace {

struct Args {
  std::string input = "data/demo_3scenes.mp4";
  std::string output = "output/final_cut.mp4";
  std::string model = "models/yolov8n.rknn";
  std::string face_model = "models/yunet.onnx";
  std::string pose_model = "models/yolov8n-pose.onnx";
  std::string shot_model = "models/transnetv2.onnx";
  std::string report;
  float conf = 0.30f;
  double analysis_fps = 6.0;
  int core_mask = 0;
  bool dump_buckets = false;
  double aes_weight = 0.38;
  double shot_fps = 8.0;
  int bench = 0;
  std::string bench_json;
  std::string dump_dets;
  std::string compare_model;
  bool zero_copy = true;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (k == "--input") a.input = next();
    else if (k == "--output") a.output = next();
    else if (k == "--model") a.model = next();
    else if (k == "--face-model") a.face_model = next();
    else if (k == "--pose-model") a.pose_model = next();
    else if (k == "--shot-model") a.shot_model = next();
    else if (k == "--report") a.report = next();
    else if (k == "--conf") a.conf = std::atof(next().c_str());
    else if (k == "--analysis-fps") a.analysis_fps = std::atof(next().c_str());
    else if (k == "--core-mask") a.core_mask = std::atoi(next().c_str());
    else if (k == "--dump-buckets") a.dump_buckets = true;
    else if (k == "--aes-weight") a.aes_weight = std::atof(next().c_str());
    else if (k == "--shot-fps") a.shot_fps = std::atof(next().c_str());
    else if (k == "--bench") a.bench = std::atoi(next().c_str());
    else if (k == "--bench-json") a.bench_json = next();
    else if (k == "--dump-dets") a.dump_dets = next();
    else if (k == "--compare-model") a.compare_model = next();
    else if (k == "--no-zero-copy") a.zero_copy = false;
    else if (k == "-h" || k == "--help") {
      printf(
          "usage: auto_editor --input VID --output MP4 [options]\n"
          "  --model yolov8n.rknn           检测（NPU）\n"
          "  --face-model yunet.onnx        人脸（CPU，可选）\n"
          "  --pose-model yolov8n-pose.onnx 姿态（CPU，可选）\n"
          "  --shot-model transnetv2.onnx   镜头切分（CPU，可选；缺失退回直方图）\n"
          "  --analysis-fps 6 --conf 0.3 --core-mask 0 --aes-weight 0.38\n"
          "  --dump-buckets                 打印每秒打分\n"
          "  --bench N                      只跑检测，输出 N 次延迟分位数\n"
          "  --bench-json PATH              把 bench 结果写成 JSON\n"
          "  --dump-dets PATH               把最后一帧检测结果写成 JSON\n"
          "  --compare-model PATH           与 --model 逐帧对比（量化前后一致性）\n"
          "  --no-zero-copy                 关闭 RKNN I/O 零拷贝，做 A/B 对比\n");
      std::exit(0);
    }
  }
  return a;
}

double laplacian_var(const cv::Mat& gray) {
  cv::Mat lap;
  cv::Laplacian(gray, lap, CV_64F);
  cv::Scalar m, s;
  cv::meanStdDev(lap, m, s);
  return s.val[0] * s.val[0];
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += ch;
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out += ch;
    }
  }
  return out;
}

// 只跑检测：预热量、分位数、有效帧率。查"检测到底多快"比跑整条流水线省事。
int run_bench(const Args& a, const ed::VideoInfo& info) {
  ed::YoloV8 det(a.model, a.conf, 0.45f, 640, a.core_mask, 100, a.zero_copy);
  cv::VideoCapture cap(a.input);
  if (!cap.isOpened()) {
    fprintf(stderr, "cannot open video: %s\n", a.input.c_str());
    return 1;
  }

  std::vector<double> ms;
  std::vector<ed::DetectTiming> timings;
  std::vector<ed::Box> last_boxes;
  cv::Mat frame;
  int warmup = 3;
  size_t last_dets = 0;
  const ed::SysSnapshot sys0 = ed::sys_snapshot();
  const double wall0 = static_cast<double>(cv::getTickCount());
  while (static_cast<int>(ms.size()) < a.bench) {
    if (!cap.read(frame) || frame.empty()) {
      cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      if (!cap.read(frame) || frame.empty()) break;  // 单帧视频，别死循环
    }
    auto dets = det.detect(frame);
    last_boxes = dets;
    last_dets = dets.size();
    if (warmup > 0) {
      --warmup;
      continue;
    }
    ms.push_back(det.last_ms());
    timings.push_back(det.timing());
  }

  if (ms.empty()) {
    fprintf(stderr, "bench got no frame\n");
    return 1;
  }
  const double wall_s = (static_cast<double>(cv::getTickCount()) - wall0) / cv::getTickFrequency();
  const ed::SysSnapshot sys1 = ed::sys_snapshot();
  const double cpu_util = wall_s > 0.0 ? (sys1.cpu_ms - sys0.cpu_ms) / (wall_s * 1000.0) * 100.0 : 0.0;

  auto stage = [&](double ed::DetectTiming::*field) {
    std::vector<double> v;
    v.reserve(timings.size());
    for (const auto& t : timings) v.push_back(t.*field);
    return ed::percentile(v, 50);
  };

  printf("[bench] %s  %dx%d @%.1ffps  %zu runs\n", a.input.c_str(), info.width, info.height,
         info.fps, ms.size());
  printf("[bench] detect p50=%.1fms p95=%.1fms mean=%.1fms  (%.1f fps)  last=%zu dets\n",
         ed::percentile(ms, 50), ed::percentile(ms, 95), ed::mean(ms),
         1000.0 / std::max(1e-6, ed::mean(ms)), last_dets);
  printf("[bench] stages p50: pre=%.1f input=%.1f npu=%.1f out=%.1f decode=%.1f nms=%.1f\n",
         stage(&ed::DetectTiming::preprocess_ms), stage(&ed::DetectTiming::input_ms),
         stage(&ed::DetectTiming::run_ms), stage(&ed::DetectTiming::output_ms),
         stage(&ed::DetectTiming::decode_ms), stage(&ed::DetectTiming::nms_ms));
  printf("[bench] system: rss=%.0fMB hwm=%.0fMB cpu=%.0f%% temp=%.1fC npu=%.0fMHz cpu=%.0fMHz\n",
         sys1.rss_mb, sys1.hwm_mb, cpu_util, sys1.temp_c, sys1.npu_mhz, sys1.cpu_mhz);

  if (!a.bench_json.empty()) {
    std::ofstream rep(a.bench_json);
    rep << std::fixed << std::setprecision(2)
        << "{\n"
        << "  \"input\": \"" << json_escape(a.input) << "\",\n"
        << "  \"model\": \"" << json_escape(a.model) << "\",\n"
        << "  \"core_mask\": " << a.core_mask << ",\n"
        << "  \"zero_copy\": " << (det.zero_copy() ? "true" : "false") << ",\n"
        << "  \"runs\": " << ms.size() << ",\n"
        << "  \"detect_ms\": {\"p50\": " << ed::percentile(ms, 50)
        << ", \"p95\": " << ed::percentile(ms, 95) << ", \"mean\": " << ed::mean(ms) << "},\n"
        << "  \"stages_p50_ms\": {"
        << "\"preprocess\": " << stage(&ed::DetectTiming::preprocess_ms)
        << ", \"input\": " << stage(&ed::DetectTiming::input_ms)
        << ", \"npu_run\": " << stage(&ed::DetectTiming::run_ms)
        << ", \"outputs_get\": " << stage(&ed::DetectTiming::output_ms)
        << ", \"decode\": " << stage(&ed::DetectTiming::decode_ms)
        << ", \"nms\": " << stage(&ed::DetectTiming::nms_ms) << "},\n"
        << "  \"system\": {\"rss_mb\": " << sys1.rss_mb << ", \"hwm_mb\": " << sys1.hwm_mb
        << ", \"cpu_util_pct\": " << cpu_util << ", \"temp_c\": " << sys1.temp_c
        << ", \"npu_mhz\": " << sys1.npu_mhz << ", \"cpu_mhz\": " << sys1.cpu_mhz << "},\n"
        << "  \"last_dets\": " << last_dets << "\n"
        << "}\n";
  }
  if (!a.dump_dets.empty()) {
    std::ofstream d(a.dump_dets);
    d << std::fixed << std::setprecision(4) << "[";
    for (size_t i = 0; i < last_boxes.size(); ++i) {
      const auto& b = last_boxes[i];
      d << (i ? ", " : "") << "{\"x1\": " << b.x1 << ", \"y1\": " << b.y1
        << ", \"x2\": " << b.x2 << ", \"y2\": " << b.y2 << ", \"score\": " << b.score
        << ", \"cls\": " << b.cls << "}";
    }
    d << "]\n";
  }
  return 0;
}

float iou_box(const ed::Box& a, const ed::Box& b) {
  const float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
  return inter / (a.area() + b.area() - inter + 1e-9f);
}

struct CompareStats {
  long frames = 0;
  long ref_n = 0, test_n = 0, matched = 0;
  long ref_person = 0, test_person = 0, matched_person = 0;
  double iou_sum = 0.0, score_abs = 0.0, center_err = 0.0;
};

// 没有标注数据时，用 FP16 当参考：同类别、IoU≥0.5 的贪心匹配。
// 指标不是 mAP，只能说明"量化后框和分数变了多少"，写报告时要讲清楚。
int run_compare(const Args& a, const ed::VideoInfo& info) {
  ed::YoloV8 ref(a.model, a.conf, 0.45f, 640, a.core_mask, 100, a.zero_copy);
  ed::YoloV8 q(a.compare_model, a.conf, 0.45f, 640, a.core_mask, 100, a.zero_copy);
  cv::VideoCapture cap(a.input);
  if (!cap.isOpened()) {
    fprintf(stderr, "cannot open video: %s\n", a.input.c_str());
    return 1;
  }

  CompareStats st;
  std::vector<double> ref_ms, q_ms;
  cv::Mat frame;
  int warmup = 3;
  while (st.frames < a.bench) {
    if (!cap.read(frame) || frame.empty()) {
      cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      if (!cap.read(frame) || frame.empty()) break;
    }
    const auto r = ref.detect(frame);
    const auto t = q.detect(frame);
    if (warmup > 0) {
      --warmup;
      continue;
    }
    ref_ms.push_back(ref.last_ms());
    q_ms.push_back(q.last_ms());
    ++st.frames;
    st.ref_n += static_cast<long>(r.size());
    st.test_n += static_cast<long>(t.size());

    std::vector<char> used(t.size(), 0);
    for (const auto& rb : r) {
      float best_iou = 0.5f;
      int best = -1;
      for (size_t j = 0; j < t.size(); ++j) {
        if (used[j] || t[j].cls != rb.cls) continue;
        const float v = iou_box(rb, t[j]);
        if (v >= best_iou) {
          best_iou = v;
          best = static_cast<int>(j);
        }
      }
      if (rb.cls == 0) ++st.ref_person;
      if (best < 0) continue;
      used[best] = 1;
      ++st.matched;
      st.iou_sum += best_iou;
      st.score_abs += std::abs(rb.score - t[best].score);
      st.center_err += std::hypot((rb.cx() - t[best].cx()), (rb.cy() - t[best].cy()));
      if (rb.cls == 0) ++st.matched_person;
    }
    for (const auto& tb : t)
      if (tb.cls == 0) ++st.test_person;
  }

  if (st.frames == 0) {
    fprintf(stderr, "compare got no frame\n");
    return 1;
  }
  const double recall = st.ref_n ? static_cast<double>(st.matched) / st.ref_n : 0.0;
  const double precision = st.test_n ? static_cast<double>(st.matched) / st.test_n : 0.0;
  const double person_recall =
      st.ref_person ? static_cast<double>(st.matched_person) / st.ref_person : 0.0;
  const double mean_iou = st.matched ? st.iou_sum / st.matched : 0.0;
  const double score_mae = st.matched ? st.score_abs / st.matched : 0.0;
  const double center_err = st.matched ? st.center_err / st.matched : 0.0;

  printf("[compare] %s  %dx%d @%.1ffps  %ld frames\n", a.input.c_str(), info.width, info.height,
         info.fps, st.frames);
  printf("[compare] ref=%s\ntest=%s\n", a.model.c_str(), a.compare_model.c_str());
  printf("[compare] dets ref=%ld test=%ld matched=%ld recall=%.3f precision=%.3f mean_iou=%.3f\n",
         st.ref_n, st.test_n, st.matched, recall, precision, mean_iou);
  printf("[compare] person ref=%ld test=%ld matched=%ld recall=%.3f\n", st.ref_person,
         st.test_person, st.matched_person, person_recall);
  printf("[compare] score_mae=%.4f center_err_px=%.2f\n", score_mae, center_err);
  printf("[compare] latency ref p50=%.1fms test p50=%.1fms\n", ed::percentile(ref_ms, 50),
         ed::percentile(q_ms, 50));

  if (!a.bench_json.empty()) {
    std::ofstream rep(a.bench_json);
    rep << std::fixed << std::setprecision(4)
        << "{\n"
        << "  \"input\": \"" << json_escape(a.input) << "\",\n"
        << "  \"ref_model\": \"" << json_escape(a.model) << "\",\n"
        << "  \"test_model\": \"" << json_escape(a.compare_model) << "\",\n"
        << "  \"zero_copy\": " << (ref.zero_copy() ? "true" : "false") << ",\n"
        << "  \"frames\": " << st.frames << ",\n"
        << "  \"ref_dets\": " << st.ref_n << ", \"test_dets\": " << st.test_n
        << ", \"matched\": " << st.matched << ",\n"
        << "  \"recall\": " << recall << ", \"precision\": " << precision
        << ", \"person_recall\": " << person_recall << ",\n"
        << "  \"mean_iou\": " << mean_iou << ", \"score_mae\": " << score_mae
        << ", \"center_err_px\": " << center_err << ",\n"
        << "  \"latency_ms\": {\"ref_p50\": " << ed::percentile(ref_ms, 50)
        << ", \"test_p50\": " << ed::percentile(q_ms, 50) << "}\n"
        << "}\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // GMC 里的 RANSAC 用了全局随机数，固定种子保证同一素材多次运行结果一致
  cv::setRNGSeed(0);
  const Args a = parse(argc, argv);

  const ed::VideoInfo info = ed::probe(a.input);
  if (info.width == 0) {
    fprintf(stderr, "cannot open video: %s\n", a.input.c_str());
    return 1;
  }
  if (a.bench > 0) {
    return a.compare_model.empty() ? run_bench(a, info) : run_compare(a, info);
  }

  const auto out_parent = std::filesystem::path(a.output).parent_path();
  if (!out_parent.empty()) std::filesystem::create_directories(out_parent);
  printf("[editor] %s  %.1fs  %dx%d @%.1ffps\n", a.input.c_str(), info.duration,
         info.width, info.height, info.fps);

  // ---- 模块初始化（人脸/姿态/切分缺失时自动降级）
  ed::YoloV8 det(a.model, a.conf, 0.45f, 640, a.core_mask, 100, a.zero_copy);
  ed::BoTSORT tracker;
  ed::FaceDetector face(a.face_model);
  ed::PoseEstimator pose(a.pose_model);
  ed::ShotDetector shot_model(a.shot_model);
  printf("[modules] detect=rknn  face=%s  pose=%s  shot=%s\n",
         face.ready() ? "yunet" : "off", pose.ready() ? "yolov8-pose" : "off",
         shot_model.using_transnet() ? "transnetv2" : "histogram");

  // ---- 镜头切分（预扫描）：边界用于约束选段，避免成片跨硬切
  const double t_shot = static_cast<double>(cv::getTickCount());
  const auto boundaries =
      shot_model.detect(a.input, info.fps, info.width, info.height, a.shot_fps);
  const double shot_sec =
      (static_cast<double>(cv::getTickCount()) - t_shot) / cv::getTickFrequency();
  printf("[shot] %zu boundaries in %.2fs:", boundaries.size(), shot_sec);
  for (size_t i = 0; i < boundaries.size() && i < 12; ++i) printf(" %.2f", boundaries[i]);
  printf("\n");

  const double stride_d =
      std::max(1.0, std::round(info.fps / std::max(1.0, a.analysis_fps)));
  const int stride = static_cast<int>(stride_d);
  const double analysis_rate = info.fps / stride;

  ed::HighlightPicker picker(analysis_rate, 1.0, 3.0, 8.0, 0.45, a.aes_weight);
  picker.set_debug(a.dump_buckets);
  picker.set_boundaries(boundaries);

  std::vector<double> detect_ms, face_ms, pose_ms;
  long analyzed = 0, detect_calls = 0, reid_calls = 0, aes_calls = 0, face_calls = 0,
       pose_calls = 0;

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

    // 人脸：约 1fps，对画面上最大的 2 个人体框内做人脸检测（裁人再检测更稳）
    if (face.ready() && idx % (stride * 6) == 0) {
      const double tf = static_cast<double>(cv::getTickCount());
      std::vector<const ed::Track*> persons;
      for (const auto& tr : tracks) {
        if (tr.cls == 0) persons.push_back(&tr);
      }
      std::sort(persons.begin(), persons.end(), [](const ed::Track* a, const ed::Track* b) {
        const auto ra = a->kf.xyxy(), rb = b->kf.xyxy();
        return ra.width * ra.height > rb.width * rb.height;
      });
      for (size_t k = 0; k < persons.size() && k < 2; ++k) {
        const auto faces = face.detect_in_box(frame, persons[k]->kf.xyxy());
        float best_frontal = -1.f;
        for (const auto& f : faces) best_frontal = std::max(best_frontal, f.frontal);
        if (a.dump_buckets) {
          printf("[face] t=%.1fs tid=%d faces=%zu", t, persons[k]->tid, faces.size());
          for (const auto& f : faces) printf(" score=%.2f frontal=%.2f", f.score, f.frontal);
          printf("\n");
        }
        if (best_frontal >= 0.f) picker.add_face(t, persons[k]->tid, best_frontal);
      }
      face_ms.push_back((static_cast<double>(cv::getTickCount()) - tf) /
                        cv::getTickFrequency() * 1000.0);
      ++face_calls;
    }

    // 姿态：约 2fps，只对画面上最大的那个人跑（控制 CPU 开销）
    if (pose.ready() && idx % (stride * 3) == 0) {
      const ed::Track* biggest = nullptr;
      float best_area = 0.f;
      for (const auto& tr : tracks) {
        if (tr.cls != 0) continue;
        const auto r = tr.kf.xyxy();
        if (r.width * r.height > best_area) {
          best_area = r.width * r.height;
          biggest = &tr;
        }
      }
      if (biggest != nullptr) {
        const double tp = static_cast<double>(cv::getTickCount());
        const auto pr = pose.estimate(frame, biggest->kf.xyxy());
        if (pr.ok) picker.add_framing(biggest->tid, pr.framing);
        pose_ms.push_back((static_cast<double>(cv::getTickCount()) - tp) /
                          cv::getTickFrequency() * 1000.0);
        ++pose_calls;
      }
    }

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
    if (!ed::cut_segment(a.input, segments[i].start, segments[i].duration(), part.str(), enc,
                         err)) {
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
  const ed::SysSnapshot sys_end = ed::sys_snapshot();

  printf("\n=== edit report ===\n");
  printf("video          : %.1fs %dx%d @%.1ffps\n", info.duration, info.width, info.height,
         info.fps);
  printf("encode         : %s\n", enc.c_str());
  printf("detector       : rknn-npu  p50=%.1fms p95=%.1fms\n", ed::percentile(detect_ms, 50),
         ed::percentile(detect_ms, 95));
  if (!face_ms.empty())
    printf("face (yunet)   : %ld calls, p50=%.1fms p95=%.1fms\n", face_calls,
           ed::percentile(face_ms, 50), ed::percentile(face_ms, 95));
  if (!pose_ms.empty())
    printf("pose (yolov8)  : %ld calls, p50=%.1fms p95=%.1fms\n", pose_calls,
           ed::percentile(pose_ms, 50), ed::percentile(pose_ms, 95));
  printf("shots          : %zu boundaries (%.2fs pre-pass)\n", boundaries.size(), shot_sec);
  printf("analysis       : %ld frames @ %.1ffps, %.1fs (detect %ld, reid %ld, aes %ld)\n",
         analyzed, analysis_rate, pass1, detect_calls, reid_calls, aes_calls);
  if (protag) {
    printf("protagonist    : tid=%d frames=%d score=%.3f salience=%.3f face_hits=%d "
           "framing=%.3f\n",
           protag->tid, protag->frames, protag->protagonist_score(), protag->mean_salience(),
           protag->face_hits, protag->mean_framing());
  }
  printf("segments       : ");
  for (const auto& s : segments) printf("(%.1f-%.1f %.2f) ", s.start, s.end, s.score);
  printf("\noutput         : %.1fs in %.1fs\n", out_info.duration, pass2);
  printf("total          : %.1fs (%.2fx realtime)\n", total,
         info.duration / std::max(total, 1e-6));

  if (!a.report.empty()) {
    std::ofstream rep(a.report);
    rep << std::fixed << std::setprecision(3) << "{\n"
        << "  \"input\": \"" << a.input << "\",\n"
        << "  \"output\": \"" << a.output << "\",\n"
        << "  \"video\": {\"width\": " << info.width << ", \"height\": " << info.height
        << ", \"fps\": " << info.fps << ", \"duration\": " << info.duration << "},\n"
        << "  \"encode_backend\": \"" << enc << "\",\n"
        << "  \"model\": \"" << json_escape(a.model) << "\",\n"
        << "  \"zero_copy\": " << (det.zero_copy() ? "true" : "false") << ",\n"
        << "  \"modules\": {\"detect\": \"rknn\", \"face\": \""
        << (face.ready() ? "yunet" : "off") << "\", \"pose\": \""
        << (pose.ready() ? "yolov8-pose" : "off") << "\", \"shot\": \""
        << (shot_model.using_transnet() ? "transnetv2" : "histogram") << "\"},\n"
        << "  \"shots\": " << boundaries.size() << ",\n"
        << "  \"analysis_fps\": " << analysis_rate << ",\n"
        << "  \"analyzed_frames\": " << analyzed << ",\n"
        << "  \"detect_calls\": " << detect_calls << ",\n"
        << "  \"face_calls\": " << face_calls << ",\n"
        << "  \"pose_calls\": " << pose_calls << ",\n"
        << "  \"reid_calls\": " << reid_calls << ",\n"
        << "  \"aesthetic_calls\": " << aes_calls << ",\n"
        << "  \"detect_ms_p50\": " << ed::percentile(detect_ms, 50) << ",\n"
        << "  \"detect_ms_p95\": " << ed::percentile(detect_ms, 95) << ",\n"
        << "  \"face_ms_p50\": " << ed::percentile(face_ms, 50) << ",\n"
        << "  \"face_ms_p95\": " << ed::percentile(face_ms, 95) << ",\n"
        << "  \"pose_ms_p50\": " << ed::percentile(pose_ms, 50) << ",\n"
        << "  \"pose_ms_p95\": " << ed::percentile(pose_ms, 95) << ",\n"
        << "  \"pass1_seconds\": " << pass1 << ",\n"
        << "  \"pass2_seconds\": " << pass2 << ",\n"
        << "  \"output_duration\": " << out_info.duration << ",\n"
        << "  \"speedup_vs_realtime\": " << info.duration / std::max(total, 1e-6) << ",\n"
        << "  \"process\": {\"rss_mb\": " << sys_end.rss_mb << ", \"hwm_mb\": " << sys_end.hwm_mb
        << ", \"cpu_total_ms\": " << sys_end.cpu_ms << ", \"temp_c\": " << sys_end.temp_c
        << ", \"npu_mhz\": " << sys_end.npu_mhz << "},\n";
    if (protag) {
      rep << "  \"protagonist\": {\"track_id\": " << protag->tid << ", \"frames\": "
          << protag->frames << ", \"score\": " << protag->protagonist_score()
          << ", \"face_hits\": " << protag->face_hits << ", \"framing\": "
          << protag->mean_framing() << "},\n";
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
