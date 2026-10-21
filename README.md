# RK3588 端侧自动剪辑

给一段视频，自动挑出最值得看的一段，剪成 5～8 秒的成片。模型和编码都在 Orange Pi 5 Pro（RK3588S）上跑，素材不出设备。

![tests](https://github.com/Muse-ztl/rk3588-auto-editor/actions/workflows/ci.yml/badge.svg)

## 能做什么

- 检测、跟踪人物，整段看完再判断谁是主角，不是"出现最久的人"
- 除了时长，还会看人物显著性、清晰度、构图和正脸
- 先做镜头切分，再在单个镜头内部挑窗口，成片不会从硬切中间开始
- RKNN 输入输出有零拷贝路径；带 INT8 转换和逐帧一致性对比脚本
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

转换脚本把 `/255` 写进 RKNN 的 `std_values`，之前用旧脚本转的 std=1 模型需要重转。

可选：INT8 只需要一批视频帧做校准。

```bash
python scripts/make_calib.py --video 你的素材.mp4 --out data/calib --num 300
python scripts/convert_yolo_rknn.py --onnx yolov8n.onnx --quantize \
  --dataset data/calib/dataset.txt --out models/yolov8n_int8.rknn
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

`--bench 100` 只跑检测，打印 p50 / p95、各阶段耗时和板端频率/温度；`--bench-json` 落盘。
人脸、姿态、切分模型都可以不给，缺了会自动降级。

## 实测

Orange Pi 5 Pro（RK3588S，16GB），单核 NPU，16s 1080p 实拍素材，`--bench 200`：

| | FP16 原版 | FP16 + 零拷贝 | INT8 |
|---|---|---|---|
| 检测 p50 | 91.7 ms | 67.4 ms | **46.4 ms** |
| 检测 p95 | 130.5 ms | 111.9 ms | **57.1 ms** |
| 模型大小 | 7.6 MB | 7.6 MB | 4.3 MB |
| 进程 RSS | 188 MB | 188 MB | 169 MB |

FP16 + 零拷贝是默认路径。12s 三镜头合成素材全流程 15.3s、选中 8.0–12.0s；
16s 实拍全流程 42.5s、选中 5.0–12.0s。检测 p50 在完整流水线里约 70ms。

INT8 检测层对 FP16 的 person recall 是 97.9%（IoU 0.5），但框有 1–5px 抖动，
实拍素材的选段会从 5–12s 偏到 4–11s，所以**没有设成默认**。量化前后对比、失败尝试和
选段偏移的分析都在 [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md)。

## 目录

```
cpp/                  板端 C++ 实现
  src/                detector / tracker / highlight / shot / face / pose / video_io
  tests/              x86 单测（不依赖 RKNN 和 ONNX Runtime）
scripts/              x86 侧模型转换、校准集、演示素材生成
assets/               实测 report 和 bench JSON
docs/OPTIMIZATION.md  零拷贝/INT8 的前后对比与失败记录
docs/NOTES.md         实现细节
```

## 限制

- 只处理单机位、无音频素材，出片时音轨直接丢掉
- 演示素材 `data/demo_3scenes.mp4` 是脚本生成的，不是实拍
- 姿态用 YOLOv8n-pose，人脸用 YuNet，外观特征默认 HSV 直方图
- INT8 默认关闭：检测更快，但框抖动会传导到跟踪和选段，见 `docs/OPTIMIZATION.md`
- 零拷贝和 INT8 都只在这块 RK3588S 上验证过，换平台要重新测

## 更多

两遍处理、选段权重、零拷贝和量化实验分别在 `docs/NOTES.md`、`docs/OPTIMIZATION.md`。
