# RK3588 端侧自动剪辑

给一段视频，自动挑出最值得看的一段，剪成 5～8 秒的成片。模型和编码都在 Orange Pi 5 Pro（RK3588S）上跑，素材不出设备。

![tests](https://github.com/Muse-ztl/rk3588-auto-editor/actions/workflows/ci.yml/badge.svg)

## 能做什么

- 检测、跟踪人物，整段看完再判断谁是主角，不是"出现最久的人"
- 除了时长，还会看人物显著性、清晰度、构图和正脸
- 先做镜头切分，再在单个镜头内部挑窗口，成片不会从硬切中间开始
- FFmpeg 切割、拼接，输出 H.264

## 跑起来

x86 上准备模型，只需要做一次：

```bash
python3 -m venv ~/rknn_env && source ~/rknn_env/bin/activate
pip install rknn-toolkit2==2.3.2 "setuptools<81" "onnx==1.16.1" "numpy<2"
python scripts/convert_yolo_rknn.py --onnx yolov8n.onnx --out models/yolov8n.rknn

pip install ultralytics transnetv2-pytorch torch --index-url https://download.pytorch.org/whl/cpu
python scripts/export_aux_models.py --out models
```

板端编译和运行：

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

./build/auto_editor \
  --input ../data/demo_3scenes.mp4 --output ../output/final_cut.mp4 \
  --model ../models/yolov8n.rknn \
  --face-model ../models/yunet.onnx \
  --pose-model ../models/yolov8n-pose.onnx \
  --shot-model ../models/transnetv2.onnx \
  --analysis-fps 6 --core-mask 1 --report ../output/report.json
```

`--bench 100` 只跑检测，打印 p50 / p95 和有效 fps。人脸、姿态、切分模型都可以不给，缺了会自动降级。

## 实测

Orange Pi 5 Pro（RK3588S，16GB），单核 NPU，分析采样 6fps：

| | 12s 三镜头（合成） | 16s 1080p（实拍） |
|---|---|---|
| 检测 p50 / p95 | 89 / 122 ms | 85 / 99 ms |
| 人脸 p50 | 110 ms | 70 ms |
| 姿态 p50 | 155 ms | 172 ms |
| 分析 pass | 15.7 s | 36.7 s |
| 出片 | 0.8 s | 6.8 s |
| 选中区间 | 8.0–12.0 s | 5.0–12.0 s |

合成素材的切点在 4.0s / 8.0s，检出 3.87s / 7.87s；实拍素材是单镜头，检出 0 个边界。
原始数据在 `assets/report_synthetic.json` 和 `assets/report_real.json`。

## 目录

```
cpp/             板端 C++ 实现
  src/           detector / tracker / highlight / shot / face / pose / video_io
  tests/         x86 单测（不依赖 RKNN 和 ONNX Runtime）
scripts/         x86 侧模型转换和演示素材生成
assets/          实测 report
docs/NOTES.md    实现细节和踩坑
```

## 限制

- 只处理单机位、无音频素材，出片时音轨直接丢掉
- 演示素材 `data/demo_3scenes.mp4` 是脚本生成的，不是实拍
- 姿态用 YOLOv8n-pose，人脸用 YuNet，外观特征默认 HSV 直方图
- 检测是 FP16；`convert_yolo_rknn.py --quantize` 支持 INT8，但仓库里没有校准集

## 更多

权重、两遍处理、选段细节和测量口径都在 `docs/NOTES.md`。
