# 端侧视频 AI 自动剪辑（C++ / RK3588S）

在 Orange Pi 5 Pro（RK3588S，6 TOPS NPU）上离线完成「看素材 → 找主角 → 剪成片」，
**全链路 C++**：

- **检测**：YOLOv8n 跑 RKNN/NPU，输出 DFL 三尺度原始张量，自己在 C++ 里解码 + NMS
- **跟踪**：BoT-SORT，8 维 Kalman + IoU/外观两级代价 + ORB 全局运动补偿
- **镜头切分**：TransNetV2（100 帧窗口 / 27×48 输入），选段不跨硬切点
- **姿态**：YOLOv8n-pose 17 关键点，判断构图（头部位置 + 关键点完整度）
- **人脸**：YuNet（ONNX Runtime 自解码），5 点几何判断正脸程度
- **美学打分**：清晰度、色彩、曝光、对比度、构图（三分法 + 主体占比）
- **剪辑决策**：主角识别 → 1s 桶显著性 × 画质 × 正脸 → 镜头内滑窗选段 → FFmpeg 切割 + 拼接

整条链路不依赖云服务，素材不出设备。

## 实测（Orange Pi 5 Pro / RK3588S）

**A. 合成素材**（12s 960×540，三段硬切，切点 4.0s / 8.0s）

| 项目 | 数值 |
|---|---|
| 镜头切分 | 检出 **2 个边界：3.87s / 7.87s**（8fps 采样，误差 <0.13s），预扫描 2.7s |
| 检测 | p50 90 ms |
| 选中片段 | **8.0–12.0s**（正好在第三个镜头内，不跨切） |

**B. 真实素材**（16.2s 1920×1080 30fps，山路徒步，单镜头）

| 阶段 | 位置 | 耗时 |
|---|---|---|
| 检测 YOLOv8n | RKNN / NPU（core0） | **89 ms**（p50） |
| 人脸 YuNet | ONNX Runtime / CPU | **68 ms**（17 次调用，约 1fps） |
| 姿态 YOLOv8n-pose | ONNX Runtime / CPU | **169 ms**（28 次调用，约 2fps） |
| 镜头切分 TransNetV2 | ONNX Runtime / CPU | 7.2 s 预扫描（0 个边界，符合单镜头） |
| 分析 pass | — | 36.5 s（97 帧 @6fps） |
| 出片 pass | — | 6.6 s（7s 成片） |
| 全流程 | — | 43.2 s |

主角：track 13，曝光 38 帧，姿态构图分 0.586。人脸在 13–16s 检出（正脸程度 0.13–0.40，
人物侧身回头），因此该段拿到正脸加分但仍未超过行走段的显著性，最终选中 5.0–12.0s。

原始数据：`assets/report_synthetic.json`、`assets/report_real.json`。

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
            ort_model.h shot.h face.h pose.h
  src/      detector.cpp    RKNN 推理 + DFL 解码 + 分类 NMS
            tracker.cpp     8 维 Kalman、代价矩阵、ORB GMC
            aesthetics.cpp  HSV 特征 + 画质打分
            highlight.cpp   主角识别 / 片段打分 / EDL
            video_io.cpp    FFmpeg 切割与拼接
            ort_model.cpp   极简 ORT 封装（单输入单输出）
            shot.cpp        TransNetV2 镜头切分（缺失时退化为直方图）
            face.cpp        YuNet 人脸检测 + 5 点正脸判定
            pose.cpp        YOLOv8-pose 姿态与构图打分
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

# 三个可选模型（缺失会自动降级：切分退回直方图、人脸/姿态关闭）
pip install ultralytics transnetv2-pytorch torch --index-url https://download.pytorch.org/whl/cpu
python scripts/export_aux_models.py --out models

# 板端
sudo cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak   # 需 2.3.2，系统自带 1.4.0
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/

cd cpp && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4

./auto_editor --input ../data/demo_3scenes.mp4 --output ../output/final_cut.mp4 \
  --model ../models/yolov8n.rknn --analysis-fps 6 --core-mask 1 \
  --report ../output/report.json

# 调参/排查
#   --dump-buckets        打印每个 1s 桶的显著性/画质/平滑分，用来解释为什么选中这一段
#   --aes-weight 0.38     片段分里画质项权重（0.38 = 0.62 显著性 + 0.38 画质）
#   --analysis-fps 6      检测采样率
```

## 已知限制

- 演示素材是合成片段（`data/sample.jpg` 三档变焦），不含真实拍摄素材。
- 姿态用的是 YOLOv8n-pose（简历里曾写 RTMPose）；人脸是 YuNet，外观特征默认 HSV 直方图。
  想完全对齐简历，可把 `pose.cpp` 的模型换成 RTMPose、`hsv_features` 换成 OSNet。
- 检测为 FP16；`scripts/convert_yolo_rknn.py --quantize` 支持 INT8，需自备校准集。
- 只处理单机位、无音频素材；出片时音频轨被丢弃。
- 外观特征默认 HSV 直方图（零依赖、<1ms）；接 OSNet 只需替换 `hsv_features` 的实现。
