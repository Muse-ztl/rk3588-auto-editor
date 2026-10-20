#include "video_io.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <opencv2/videoio.hpp>

namespace ed {
namespace {

std::string run_capture(const std::string& cmd, int* code) {
  std::string out;
  FILE* p = popen(cmd.c_str(), "r");
  if (p == nullptr) {
    if (code) *code = -1;
    return out;
  }
  char buf[512];
  while (std::fgets(buf, sizeof(buf), p) != nullptr) out += buf;
  const int rc = pclose(p);
  if (code) *code = rc;
  return out;
}

std::string quote(const std::string& s) { return "\"" + s + "\""; }

}  // namespace

VideoInfo probe(const std::string& path) {
  VideoInfo info;
  info.path = path;
  cv::VideoCapture cap(path);
  if (!cap.isOpened()) return info;
  info.fps = cap.get(cv::CAP_PROP_FPS);
  if (info.fps <= 0) info.fps = 25.0;
  info.n_frames = static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  info.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  info.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  info.duration = info.fps > 0 ? info.n_frames / info.fps : 0.0;
  cap.release();
  return info;
}

std::string pick_encoder() {
  int code = 0;
  const std::string out = run_capture("ffmpeg -hide_banner -encoders 2>/dev/null", &code);
  if (out.find("h264_rkmpp") != std::string::npos) return "h264_rkmpp";
  return "libx264";
}

bool cut_segment(const std::string& input, double start, double duration,
                 const std::string& out, const std::string& encoder, std::string& err) {
  std::ostringstream cmd;
  cmd << "ffmpeg -hide_banner -loglevel error -y -ss " << start << " -t " << duration
      << " -i " << quote(input) << " -an ";
  if (encoder == "h264_rkmpp") {
    cmd << "-c:v h264_rkmpp -b:v 8M ";
  } else {
    cmd << "-c:v libx264 -preset veryfast -crf 22 ";
  }
  cmd << "-pix_fmt yuv420p " << quote(out);

  int code = 0;
  const std::string log = run_capture(cmd.str() + " 2>&1", &code);
  if (code != 0) {
    err = log.substr(0, 400);
    return false;
  }
  return true;
}

bool concat_segments(const std::vector<std::string>& parts, const std::string& out,
                     std::string& err) {
  const std::string list = out + ".list.txt";
  {
    std::ofstream f(list);
    // concat 会按 list 文件所在目录解析相对路径，这里统一写绝对路径
    for (const auto& p : parts) {
      f << "file '" << std::filesystem::absolute(p).string() << "'\n";
    }
  }
  std::ostringstream cmd;
  cmd << "ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i " << quote(list)
      << " -c copy " << quote(out);

  int code = 0;
  const std::string log = run_capture(cmd.str() + " 2>&1", &code);
  std::remove(list.c_str());
  if (code != 0) {
    err = log.substr(0, 400);
    return false;
  }
  return true;
}

}  // namespace ed
