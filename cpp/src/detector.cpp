#include "detector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace ed {
namespace {

const char* kCoco80[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"};

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

const char* coco80(int idx) {
  const int n = static_cast<int>(sizeof(kCoco80) / sizeof(kCoco80[0]));
  return (idx >= 0 && idx < n) ? kCoco80[idx] : "cls";
}

const char* Box::label() const { return coco80(cls); }

YoloV8::YoloV8(const std::string& model_path, float conf, float iou, int imgsz,
               int core_mask, int max_det)
    : imgsz_(imgsz), conf_(conf), iou_(iou), max_det_(max_det) {
  FILE* fp = std::fopen(model_path.c_str(), "rb");
  if (fp == nullptr) throw std::runtime_error("cannot open rknn: " + model_path);
  std::fseek(fp, 0, SEEK_END);
  const long sz = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  std::vector<uint8_t> model(sz);
  if (std::fread(model.data(), 1, sz, fp) != static_cast<size_t>(sz)) {
    std::fclose(fp);
    throw std::runtime_error("short read on rknn model");
  }
  std::fclose(fp);

  int ret = rknn_init(&ctx_, model.data(), static_cast<uint32_t>(model.size()),
                      RKNN_FLAG_PRIOR_MEDIUM, nullptr);
  if (ret != RKNN_SUCC) throw std::runtime_error("rknn_init failed: " + std::to_string(ret));
  alive_ = true;

  if (core_mask > 0) {
    rknn_core_mask mask = RKNN_NPU_CORE_AUTO;
    switch (core_mask) {
      case 1: mask = RKNN_NPU_CORE_0; break;
      case 3: mask = RKNN_NPU_CORE_0_1; break;
      case 7: mask = RKNN_NPU_CORE_0_1_2; break;
      default: break;
    }
    rknn_set_core_mask(ctx_, mask);
  }

  rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  n_out_ = std::min<int>(io_num_.n_output, 12);
  for (int i = 0; i < n_out_; ++i) {
    out_attr_[i].index = i;
    rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr_[i], sizeof(rknn_tensor_attr));
  }
}

YoloV8::~YoloV8() {
  if (alive_) rknn_destroy(ctx_);
}

void YoloV8::preprocess(const cv::Mat& bgr, std::vector<float>& out, float& ratio,
                        float& pad_x, float& pad_y) const {
  const int h = bgr.rows, w = bgr.cols;
  ratio = std::min(static_cast<float>(imgsz_) / h, static_cast<float>(imgsz_) / w);
  const int nh = static_cast<int>(std::round(h * ratio));
  const int nw = static_cast<int>(std::round(w * ratio));
  pad_x = (imgsz_ - nw) / 2.f;
  pad_y = (imgsz_ - nh) / 2.f;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

  cv::Mat canvas(imgsz_, imgsz_, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(static_cast<int>(std::round(pad_x - 0.1f)),
                                 static_cast<int>(std::round(pad_y - 0.1f)), nw, nh)));

  cv::Mat rgb;
  cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  // RKNN 的 normalize 只接受 NHWC 源布局
  out.resize(static_cast<size_t>(imgsz_) * imgsz_ * 3);
  for (int y = 0; y < imgsz_; ++y) {
    const cv::Vec3f* row = rgb.ptr<cv::Vec3f>(y);
    for (int x = 0; x < imgsz_; ++x) {
      const size_t base = (static_cast<size_t>(y) * imgsz_ + x) * 3;
      out[base + 0] = row[x][0];
      out[base + 1] = row[x][1];
      out[base + 2] = row[x][2];
    }
  }
}

float YoloV8::at(const float* p, int c, int y, int x, int H, int W, int C, bool chw) {
  if (chw) return p[(static_cast<size_t>(c) * H + y) * W + x];
  return p[(static_cast<size_t>(y) * W + x) * C + c];
}

void YoloV8::decode_scale(const std::vector<float>& box, const float* cls, int H, int W,
                          int nc, float stride, std::vector<Box>& cands) const {
  // box: (64) = 4 边 × 16 bins 的 DFL 分布；cls: (nc, H, W) 已过 sigmoid
  const int bins = 16;
  std::vector<float> dist(4, 0.f);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int best = -1;
      float best_p = conf_;
      for (int c = 0; c < nc; ++c) {
        const float p = at(cls, c, y, x, H, W, nc, true);
        if (p > best_p) {
          best_p = p;
          best = c;
        }
      }
      if (best < 0) continue;

      for (int side = 0; side < 4; ++side) {
        float m = -1e9f;
        float vals[bins];
        for (int b = 0; b < bins; ++b) {
          vals[b] = at(box.data(), side * bins + b, y, x, H, W, 64, true);
          m = std::max(m, vals[b]);
        }
        float sum = 0.f, acc = 0.f;
        for (int b = 0; b < bins; ++b) {
          const float e = std::exp(vals[b] - m);
          sum += e;
          acc += e * b;
        }
        dist[side] = (sum > 0 ? acc / sum : 0.f) * stride;
      }

      const float cx = (x + 0.5f) * stride;
      const float cy = (y + 0.5f) * stride;
      Box b;
      b.x1 = cx - dist[0];
      b.y1 = cy - dist[1];
      b.x2 = cx + dist[2];
      b.y2 = cy + dist[3];
      b.score = best_p;
      b.cls = best;
      cands.push_back(b);
    }
  }
}

std::vector<Box> YoloV8::nms(std::vector<Box>& cands, float ratio, float pad_x,
                             float pad_y) const {
  std::vector<Box> keep;
  for (int c = 0; c < 80; ++c) {
    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    std::vector<int> idx;
    for (size_t i = 0; i < cands.size(); ++i) {
      if (cands[i].cls != c) continue;
      rects.emplace_back(cv::Rect(cv::Point(static_cast<int>(cands[i].x1),
                                            static_cast<int>(cands[i].y1)),
                                  cv::Point(static_cast<int>(cands[i].x2),
                                            static_cast<int>(cands[i].y2))));
      scores.push_back(cands[i].score);
      idx.push_back(static_cast<int>(i));
    }
    if (rects.empty()) continue;
    std::vector<int> kept;
    cv::dnn::NMSBoxes(rects, scores, conf_, iou_, kept);
    for (int k : kept) {
      Box b = cands[idx[k]];
      b.x1 = (b.x1 - pad_x) / ratio;
      b.y1 = (b.y1 - pad_y) / ratio;
      b.x2 = (b.x2 - pad_x) / ratio;
      b.y2 = (b.y2 - pad_y) / ratio;
      keep.push_back(b);
    }
  }
  std::sort(keep.begin(), keep.end(), [](const Box& a, const Box& b) {
    return a.score > b.score;
  });
  if (static_cast<int>(keep.size()) > max_det_) keep.resize(max_det_);
  return keep;
}

std::vector<Box> YoloV8::detect(const cv::Mat& bgr) {
  const double t0 = now_ms();
  std::vector<float> input;
  float ratio = 1.f, pad_x = 0.f, pad_y = 0.f;
  preprocess(bgr, input, ratio, pad_x, pad_y);

  rknn_input in{};
  in.index = 0;
  in.buf = input.data();
  in.size = static_cast<uint32_t>(input.size() * sizeof(float));
  in.type = RKNN_TENSOR_FLOAT32;
  in.fmt = RKNN_TENSOR_NHWC;
  in.pass_through = 0;
  if (rknn_inputs_set(ctx_, 1, &in) != RKNN_SUCC) throw std::runtime_error("rknn_inputs_set failed");
  if (rknn_run(ctx_, nullptr) != RKNN_SUCC) throw std::runtime_error("rknn_run failed");

  std::vector<rknn_output> outs(n_out_);
  for (int i = 0; i < n_out_; ++i) {
    outs[i].index = i;
    outs[i].want_float = 1;
    outs[i].is_prealloc = 0;
  }
  if (rknn_outputs_get(ctx_, n_out_, outs.data(), nullptr) != RKNN_SUCC) {
    throw std::runtime_error("rknn_outputs_get failed");
  }

  std::vector<Box> cands;
  cands.reserve(20000);
  // 每 3 个输出一个尺度：box(64) / cls(nc) / 置信度和(1)
  for (int i = 0; i + 2 < n_out_; i += 3) {
    const auto& ab = out_attr_[i];
    const auto& ac = out_attr_[i + 1];
    // 从 dims 判断布局：哪一维等于 64 就是通道维
    int H = 0, W = 0;
    bool chw = true;
    if (ab.dims[1] == 64) {
      chw = true;
      H = static_cast<int>(ab.dims[2]);
      W = static_cast<int>(ab.dims[3]);
    } else {
      chw = false;
      H = static_cast<int>(ab.dims[1]);
      W = static_cast<int>(ab.dims[2]);
    }
    const int nc = static_cast<int>(ac.dims[chw ? 1 : 3]);
    const float stride = static_cast<float>(imgsz_) / H;

    std::vector<float> box(outs[i].size / sizeof(float));
    std::memcpy(box.data(), outs[i].buf, outs[i].size);
    const auto* cls = static_cast<const float*>(outs[i + 1].buf);
    decode_scale(box, cls, H, W, nc, stride, cands);
  }
  rknn_outputs_release(ctx_, n_out_, outs.data());

  auto result = nms(cands, ratio, pad_x, pad_y);
  last_ms_ = now_ms() - t0;
  return result;
}

}  // namespace ed
