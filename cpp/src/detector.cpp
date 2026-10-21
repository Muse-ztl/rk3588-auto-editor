#include "detector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "dfl.h"

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
               int core_mask, int max_det, bool zero_copy)
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
  if (std::getenv("ED_DEBUG_ATTR") != nullptr) {
    rknn_tensor_attr in{};
    in.index = 0;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in, sizeof(in));
    printf("[attr] in type=%d fmt=%d size=%u dims=[%u,%u,%u,%u]\n", in.type, in.fmt, in.size,
           in.dims[0], in.dims[1], in.dims[2], in.dims[3]);
    for (int i = 0; i < n_out_; ++i) {
      printf("[attr] out%d type=%d fmt=%d size=%u dims=[%u,%u,%u,%u]\n", i, out_attr_[i].type,
             out_attr_[i].fmt, out_attr_[i].size, out_attr_[i].dims[0], out_attr_[i].dims[1],
             out_attr_[i].dims[2], out_attr_[i].dims[3]);
    }
  }

  if (zero_copy && setup_zero_copy()) {
    zero_copy_ = true;
  } else {
    zero_copy_ = false;
  }
  printf("[detector] zero-copy=%s\n", zero_copy_ ? "on" : "off");
}

bool YoloV8::setup_zero_copy() {
  in_attr_ = {};
  in_attr_.index = 0;
  if (rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr_, sizeof(in_attr_)) != RKNN_SUCC) {
    return false;
  }
  if (in_attr_.fmt != RKNN_TENSOR_NHWC) return false;
  if (in_attr_.type != RKNN_TENSOR_FLOAT16 && in_attr_.type != RKNN_TENSOR_FLOAT32) return false;

  // 输入绑成 FP16 io_mem + pass_through，CPU 侧做 /255 和 FP16 转换，
  // 让 runtime 跳过 normalize（否则 normalize 会被插进 NPU，run 慢 ~17ms）。
  in_attr_.type = RKNN_TENSOR_FLOAT16;
  in_attr_.size = static_cast<uint32_t>(imgsz_) * imgsz_ * 3 * sizeof(uint16_t);
  in_attr_.pass_through = 1;
  in_mem_ = rknn_create_mem(ctx_, in_attr_.size);
  if (in_mem_ == nullptr || rknn_set_io_mem(ctx_, in_mem_, &in_attr_) != RKNN_SUCC) {
    release_mem();
    return false;
  }

  out_mem_.assign(n_out_, nullptr);
  out_f32_.resize(n_out_);
  for (int i = 0; i < n_out_; ++i) {
    auto& a = out_attr_[i];
    if (a.fmt != RKNN_TENSOR_NCHW) {
      release_mem();
      return false;
    }
    if (a.type != RKNN_TENSOR_FLOAT16 && a.type != RKNN_TENSOR_FLOAT32) {
      release_mem();
      return false;
    }
    out_mem_[i] = rknn_create_mem(ctx_, a.size);
    if (out_mem_[i] == nullptr || rknn_set_io_mem(ctx_, out_mem_[i], &a) != RKNN_SUCC) {
      release_mem();
      return false;
    }
    const int C = static_cast<int>(a.dims[1]);
    const int H = static_cast<int>(a.dims[2]);
    const int W = static_cast<int>(a.dims[3]);
    out_f32_[i].create(C * H, W, CV_32F);
  }
  return true;
}

void YoloV8::release_mem() {
  for (auto* m : out_mem_) {
    if (m != nullptr) rknn_destroy_mem(ctx_, m);
  }
  out_mem_.clear();
  if (in_mem_ != nullptr) {
    rknn_destroy_mem(ctx_, in_mem_);
    in_mem_ = nullptr;
  }
}

YoloV8::~YoloV8() {
  if (alive_) {
    release_mem();
    rknn_destroy(ctx_);
  }
}

cv::Mat YoloV8::make_rgb(const cv::Mat& bgr, float& ratio, float& pad_x, float& pad_y) const {
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
  return rgb;
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
  float ratio = 1.f, pad_x = 0.f, pad_y = 0.f;
  const cv::Mat rgb8 = make_rgb(bgr, ratio, pad_x, pad_y);
  const double t1 = now_ms();

  std::vector<rknn_output> outs;
  if (zero_copy_ && in_mem_ != nullptr) {
    // 直接写进 RKNN 的输入内存，省掉 vector + rknn_inputs_set 的一次拷贝
    cv::Mat dst(imgsz_, imgsz_, CV_16FC3, in_mem_->virt_addr);
    rgb8.convertTo(dst, CV_16F, 1.0 / 255.0);
    rknn_mem_sync(ctx_, in_mem_, RKNN_MEMORY_SYNC_TO_DEVICE);
  } else {
    // 直接喂 uint8 NHWC：runtime 自己会做 /255 和量化，少拷 3/4 的数据
    rknn_input in{};
    in.index = 0;
    in.buf = const_cast<uint8_t*>(rgb8.data);
    in.size = static_cast<uint32_t>(rgb8.total() * rgb8.elemSize());
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    in.pass_through = 0;
    if (rknn_inputs_set(ctx_, 1, &in) != RKNN_SUCC) {
      throw std::runtime_error("rknn_inputs_set failed");
    }
  }
  const double t2 = now_ms();

  if (rknn_run(ctx_, nullptr) != RKNN_SUCC) throw std::runtime_error("rknn_run failed");
  const double t3 = now_ms();

  if (zero_copy_) {
    for (int i = 0; i < n_out_; ++i) {
      rknn_mem_sync(ctx_, out_mem_[i], RKNN_MEMORY_SYNC_FROM_DEVICE);
      const auto& a = out_attr_[i];
      const int C = static_cast<int>(a.dims[1]);
      const int H = static_cast<int>(a.dims[2]);
      const int W = static_cast<int>(a.dims[3]);
      const int stype = a.type == RKNN_TENSOR_FLOAT16 ? CV_16F : CV_32F;
      cv::Mat view(C * H, W, stype, out_mem_[i]->virt_addr);
      view.convertTo(out_f32_[i], CV_32F);
    }
  } else {
    outs.resize(n_out_);
    for (int i = 0; i < n_out_; ++i) {
      outs[i].index = i;
      outs[i].want_float = 1;
      outs[i].is_prealloc = 0;
    }
    if (rknn_outputs_get(ctx_, n_out_, outs.data(), nullptr) != RKNN_SUCC) {
      throw std::runtime_error("rknn_outputs_get failed");
    }
  }
  const double t4 = now_ms();

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

    // box 和 cls 用同一种 layout：模型来自同一次导出，不会一半 CHW 一半 NHWC
    const float* box_ptr =
        zero_copy_ ? out_f32_[i].ptr<float>() : static_cast<const float*>(outs[i].buf);
    const float* cls_ptr =
        zero_copy_ ? out_f32_[i + 1].ptr<float>() : static_cast<const float*>(outs[i + 1].buf);
    decode_dfl_scale(box_ptr, cls_ptr, H, W, nc, 16, stride, conf_, chw, chw, cands);
  }
  if (!zero_copy_) rknn_outputs_release(ctx_, n_out_, outs.data());
  const double t5 = now_ms();

  auto result = nms(cands, ratio, pad_x, pad_y);
  const double t6 = now_ms();

  timing_.preprocess_ms = t1 - t0;
  timing_.input_ms = t2 - t1;
  timing_.run_ms = t3 - t2;
  timing_.output_ms = t4 - t3;
  timing_.decode_ms = t5 - t4;
  timing_.nms_ms = t6 - t5;
  timing_.total_ms = t6 - t0;
  last_ms_ = timing_.total_ms;
  return result;
}

}  // namespace ed
