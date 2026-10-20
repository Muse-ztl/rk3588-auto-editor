#include "face.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

namespace ed {
namespace {

constexpr int kInput = 640;  // 权重固定 640×640（也必须是 32 的倍数）

// 五点几何判断正脸程度：侧脸时眼/鼻/嘴的对称性会明显破坏
float frontal_score(const cv::Point2f pts[5]) {
  const cv::Point2f eye_r = pts[0], eye_l = pts[1], nose = pts[2];
  const cv::Point2f mouth_r = pts[3], mouth_l = pts[4];
  const float eye_dx = std::abs(eye_l.x - eye_r.x);
  if (eye_dx < 1e-3f) return 0.f;
  const float eye_dy = std::abs(eye_l.y - eye_r.y) / eye_dx;
  const float nose_mid = std::abs(nose.x - (eye_l.x + eye_r.x) * 0.5f) / eye_dx;
  const float mouth_mid = std::abs((mouth_l.x + mouth_r.x) * 0.5f - nose.x) / eye_dx;
  const float symmetry = 1.f - std::min(1.f, nose_mid * 2.f);
  const float level = 1.f - std::min(1.f, eye_dy * 3.f);
  const float mouth_ok = 1.f - std::min(1.f, mouth_mid * 2.f);
  return std::clamp(0.45f * symmetry + 0.35f * level + 0.20f * mouth_ok, 0.f, 1.f);
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
  return inter / (a.width * a.height + b.width * b.height - inter + 1e-6f);
}

}  // namespace

struct FaceDetector::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "yunet"};
  std::unique_ptr<Ort::Session> session;
  std::string input_name;
  std::vector<std::string> output_names;
  std::vector<const char*> output_ptrs;
  std::vector<int64_t> input_shape{1, 3, kInput, kInput};
};

FaceDetector::FaceDetector(const std::string& model_path, float score_thr)
    : score_thr_(score_thr) {
  if (model_path.empty()) return;
  impl_ = std::make_shared<Impl>();
  try {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    impl_->input_name = impl_->session->GetInputNameAllocated(0, alloc).get();
    for (size_t i = 0; i < impl_->session->GetOutputCount(); ++i) {
      impl_->output_names.emplace_back(impl_->session->GetOutputNameAllocated(i, alloc).get());
    }
    for (auto& n : impl_->output_names) impl_->output_ptrs.push_back(n.c_str());
  } catch (const Ort::Exception&) {
    impl_.reset();
  }
}

bool FaceDetector::ready() const { return impl_ && impl_->session; }

std::vector<Face> FaceDetector::detect_in_box(const cv::Mat& frame,
                                              const cv::Rect2f& person_box) {
  std::vector<Face> out;
  if (!ready() || frame.empty()) return out;

  // 人体框外扩 15%，避免下巴/额头被切掉
  const float mx = person_box.width * 0.15f, my = person_box.height * 0.15f;
  cv::Rect2f roi(person_box.x - mx, person_box.y - my, person_box.width + 2 * mx,
                 person_box.height + 2 * my);
  roi &= cv::Rect2f(0, 0, static_cast<float>(frame.cols), static_cast<float>(frame.rows));
  if (roi.width < 16 || roi.height < 16) return out;

  cv::Mat crop;
  cv::resize(frame(cv::Rect(static_cast<int>(roi.x), static_cast<int>(roi.y),
                            static_cast<int>(roi.width), static_cast<int>(roi.height))),
             crop, cv::Size(kInput, kInput), 0, 0, cv::INTER_LINEAR);

  // YuNet 输入：BGR、0~255、NCHW
  std::vector<float> input(static_cast<size_t>(3) * kInput * kInput);
  const int plane = kInput * kInput;
  for (int y = 0; y < kInput; ++y) {
    const cv::Vec3b* row = crop.ptr<cv::Vec3b>(y);
    for (int x = 0; x < kInput; ++x) {
      for (int c = 0; c < 3; ++c) {
        input[static_cast<size_t>(c) * plane + y * kInput + x] = row[x][c];
      }
    }
  }

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                impl_->input_shape.data(),
                                                impl_->input_shape.size());
  const char* in_name = impl_->input_name.c_str();
  std::vector<Ort::Value> outs;
  try {
    outs = impl_->session->Run(Ort::RunOptions{nullptr}, &in_name, &tensor, 1,
                               impl_->output_ptrs.data(), impl_->output_ptrs.size());
  } catch (const Ort::Exception&) {
    return out;  // 尺寸/模型不匹配时安静降级，别让整个剪辑流程挂掉
  }

  // 输出顺序：cls_8/16/32, obj_8/16/32, bbox_8/16/32, kps_8/16/32
  const int strides[3] = {8, 16, 32};
  const float inv_x = static_cast<float>(roi.width) / kInput;
  const float inv_y = static_cast<float>(roi.height) / kInput;

  std::vector<Face> cands;
  for (int si = 0; si < 3; ++si) {
    const int s = strides[si];
    const int cols = kInput / s, rows = kInput / s;
    const float* cls = outs[si].GetTensorData<float>();
    const float* obj = outs[si + 3].GetTensorData<float>();
    const float* bbox = outs[si + 6].GetTensorData<float>();
    const float* kps = outs[si + 9].GetTensorData<float>();
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const int idx = r * cols + c;
        const float score =
            std::sqrt(std::max(0.f, cls[idx]) * std::max(0.f, obj[idx]));
        if (score < score_thr_) continue;
        const float cx = (c + bbox[idx * 4 + 0]) * s;
        const float cy = (r + bbox[idx * 4 + 1]) * s;
        const float w = std::exp(bbox[idx * 4 + 2]) * s;
        const float h = std::exp(bbox[idx * 4 + 3]) * s;
        Face f;
        f.score = score;
        f.box = cv::Rect2f(roi.x + (cx - w / 2) * inv_x, roi.y + (cy - h / 2) * inv_y,
                           w * inv_x, h * inv_y);
        for (int k = 0; k < 5; ++k) {
          f.landmarks[k] =
              cv::Point2f(roi.x + (c + kps[idx * 10 + k * 2]) * s * inv_x,
                          roi.y + (r + kps[idx * 10 + k * 2 + 1]) * s * inv_y);
        }
        f.frontal = frontal_score(f.landmarks);
        cands.push_back(f);
      }
    }
  }

  std::sort(cands.begin(), cands.end(),
            [](const Face& a, const Face& b) { return a.score > b.score; });
  for (const auto& f : cands) {
    bool keep = true;
    for (const auto& k : out) {
      if (iou(f.box, k.box) > 0.3f) {
        keep = false;
        break;
      }
    }
    if (keep) out.push_back(f);
    if (out.size() >= 5) break;
  }
  return out;
}

}  // namespace ed
