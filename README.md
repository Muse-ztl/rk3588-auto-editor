# 端侧视频 AI 自动剪辑（C++ / RK3588S）

在 Orange Pi 5 Pro（RK3588S，6 TOPS NPU）上离线完成「看素材 → 找主角 → 剪成片」，
**全链路 C++**：

- **YOLOv8n** 跑 RKNN/NPU 做检测，输出 DFL 三尺度原始张量，自己在 C++ 里解码 + NMS
- **BoT-SORT**：8 维 Kalman + IoU/外观两级代价 + ORB 全局运动补偿，维持跨帧 ID
- **美学打分**：清晰度、色彩、曝光、对比度、构图（三分法 + 主体占比）
- **剪辑决策**：主角识别 → 1 秒桶显著性 × 画质 → 滑窗选段 → FFmpeg 精确切割 + 无损拼接

整条链路不依赖云服务，素材不出设备。

![demo frame](assets/cut_frame.jpg)

## 实测（Orange Pi 5 Pro / 12s 960×540 素材）

| 项目 | 数值 |
|---|---|
| 检测（YOLOv8n RKNN，NPU 单核） | p50 **90 ms** / p90 **94 ms** |
| 分析采样率 | 6 fps（可配 `--analysis-fps`） |
| 分析 pass | 12.4 s（72 帧：检测 72 次 / ReID 36 次 / 美学 6 次） |
| 出片 pass | 0.97 s（5 s 成片） |
| 选中片段 | **7.0–12.0 s**（主角占比 1.00，片段分 0.925） |
| 主角 | track_id=2，曝光 23 帧，显著性 0.574 |
| 全流程 | 13.3 s（0.90× 实时） |

原始数据见 `assets/report_demo.json`。

> 12s 素材跑 13.3s：瓶颈是检测（90ms/帧）。这个场景的瓶颈不在编码，
> 出片只花了 0.97s（5s 成片，约 5× 实时）；想再快就降采样率或上 INT8。

## 流水线

```
解码(OpenCV/FFmpeg) → 分层采样
   ├─ 每帧      : YOLOv8n(NPU) → BoT-SORT(Kalman+IoU+外观+GMC)
   ├─ 每 2 帧   : HSV 直方图外观特征
   └─ 每 12 帧  : 美学打分（清晰度/色彩/曝光/构图）
                        │
        主角识别（时长+中心度+清晰度+正脸+置信度）
                        │
        1s 桶显著性 × 画质 → 滑窗选段 → EDL
                        │
        FFmpeg 精确切割（统一编码参数）→ concat 无损拼接
```

设计取舍（和 Python 版一致，只是换了语言）：

- **分层采样**：检测最贵、画质变化最慢，所以检测 6–12fps、美学 1fps，省 5–10 倍算力。
- **主角 ≠ 出现最久**：`0.34×时长 + 0.28×中心度 + 0.20×清晰度 + 0.10×正脸 + 0.08×置信度`。
- **片段评分用显著性而非 track id**：硬切场景里 ID 会重建，按 ID 统计会抖，
  改用"该秒内人物显著性（中心度 × 主体尺寸）"跨镜头也成立。
- **滑窗选段**：阈值分组会把全片并成一段再被 `max_clip` 从 0s 砍断，恰好丢掉最精彩的部分。
- **贪心匹配**：目标数 <20 时与匈牙利结果基本一致，省掉一个依赖。

## 目录

```
cpp/
  CMakeLists.txt
  include/  types.h detector.h tracker.h aesthetics.h highlight.h video_io.h
  src/      detector.cpp    RKNN 推理 + DFL 解码 + 分类 NMS
            tracker.cpp     8 维 Kalman、代价矩阵、ORB GMC
            aesthetics.cpp  HSV 特征 + 画质打分
            highlight.cpp   主角识别 / 片段打分 / EDL
            video_io.cpp    FFmpeg 切割与拼接
            main.cpp        两 pass 主流程 + report.json
  third_party/rknn/include/rknn_api.h
scripts/                    x86 侧模型准备（Python）
  convert_yolo_rknn.py      ONNX → RKNN
  make_demo_clip.py         合成演示素材（三档变焦）
```

## 复现

```bash
# x86：YOLOv8n ONNX → RKNN（rknn-toolkit2 2.3.2）
python3 -m venv ~/rknn_env && source ~/rknn_env/bin/activate
pip install rknn-toolkit2==2.3.2 "setuptools<81" "onnx==1.16.1" "numpy<2"
python scripts/convert_yolo_rknn.py --onnx yolov8n.onnx --out models/yolov8n.rknn

# 板端
sudo cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak   # 需 2.3.2，系统自带 1.4.0
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/

cd cpp && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4

./auto_editor --input ../data/demo_3scenes.mp4 --output ../output/final_cut.mp4 \
  --model ../models/yolov8n.rknn --analysis-fps 6 --core-mask 1 \
  --report ../output/report.json
```

## 已知限制

- 演示素材是合成片段（`data/sample.jpg` 三档变焦），不含真实拍摄素材。
- 姿态/人脸模型未接入：主角打分里"正脸"恒为 0，权重位已留好。
- 检测为 FP16；`scripts/convert_yolo_rknn.py --quantize` 支持 INT8，需自备校准集。
- 只处理单机位、无音频素材；出片时音频轨被丢弃。
- 外观特征默认 HSV 直方图（零依赖、<1ms）；接 OSNet 只需替换 `hsv_features` 的实现。
