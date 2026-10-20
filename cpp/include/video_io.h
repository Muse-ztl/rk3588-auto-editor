// 视频读写：OpenCV 解码 + FFmpeg 精确切割/拼接。
#pragma once

#include <string>
#include <vector>

#include "types.h"

namespace ed {

VideoInfo probe(const std::string& path);

// 统一编码参数切一段，便于后续无损 concat
bool cut_segment(const std::string& input, double start, double duration,
                 const std::string& out, const std::string& encoder, std::string& err);

bool concat_segments(const std::vector<std::string>& parts, const std::string& out,
                     std::string& err);

// 板上有 h264_rkmpp 就用硬件编码，否则回退 libx264
std::string pick_encoder();

}  // namespace ed
