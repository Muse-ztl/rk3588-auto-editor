# 推理优化记录

这里的数字都是在 Orange Pi 5 Pro（RK3588S，16GB）上跑的，NPU runtime 2.3.2，单核 NPU（`--core-mask 1`），
输入是 `data/user_material.mp4`（1920×1080 / 30fps / 16.2s）。原始 JSON 在 `assets/`。

## 测量协议

- `--bench 200`：只跑检测，前 3 次预热，记录 200 次；p50/p95 是最近秩。
- 每次记录 `VmRSS`、进程 CPU 时间、温度、NPU/CPU 当前频率（`cpp/include/sysinfo.h`）。
- 同一台板子、同一段素材；报告里保留温度和频率，避免拿 40°C 的数字和 55°C 的数字比。
- 板子没有 root，锁不了 governor，所以 p50 会有几 ms 的波动；趋势看多轮一致的结果。

## Baseline：时间花在哪

`--no-zero-copy`，FP16 模型：

| 阶段 | p50 |
|---|---|
| letterbox + RGB | 6.5 ms |
| `rknn_inputs_set`（float32 → FP16） | 11.3 ms |
| `rknn_run` | 56.8 ms |
| `rknn_outputs_get`（FP16 → float32） | 11.6 ms |
| DFL 解码 | 4.9 ms |
| NMS | 0.0 ms |
| **合计** | **91.7 ms** |

检测本身只占 57ms，**I/O 拷贝占了 23ms**，这是第一个该动的地方。

## 零拷贝：两次尝试

### 1）输入输出都绑 `rknn_set_io_mem`（直接喂 float）

`rknn_create_mem` + `rknn_set_io_mem` 接管输入和 9 个输出，输入按 FP16/FP32 写。

结果：检测框完全一致，但 `rknn_run` 从 57ms 涨到 70–76ms。原因是输入绑成 io_mem 后，
RKNN 把 `/255` 的 normalize 挪进了 NPU，而这段本来在 `rknn_inputs_set` 里做。I/O 省了 20ms，
NPU 又吃回去 17ms，净收益只有 3–5ms。

### 2）输入 FP16 + `pass_through`，CPU 做 `/255`（最终方案）

- 模型转换时把归一化写成 `std_values=255`，校准时读原图，分布和线上一致。
- 输入内存按 FP16 创建，`pass_through=1`；C++ 里 `convertTo(CV_16F, 1/255)` 直接写进这块内存。
- 输出仍然用 io_mem，读之前 `rknn_mem_sync(FROM_DEVICE)`，FP16→float32 只在需要的输出上做。

| 阶段 | 原版 | 零拷贝 |
|---|---|---|
| 输入 | 11.3 ms | 1.1 ms |
| `rknn_run` | 56.8 ms | 41.4 ms |
| 输出 | 11.6 ms | 1.9 ms |
| DFL 解码 | 4.9 ms | 11.2 ms |
| **p50 / p95** | **91.7 / 130.5 ms** | **67.4 / 111.9 ms** |

检测框逐位一致（`--dump-dets` 对比过）。解码变慢是因为输出现在是 FP16 转 float 后再解，
多了一次转换；总体还是赚的。

## INT8 PTQ

### 校准集

- `scripts/make_calib.py` 从视频抽 300 帧，按 C++ 的 letterbox（114 padding）保存成 640×640 JPEG。
- 关键点：归一化在 RKNN 的 `std=255` 里，所以校准读 0–255 原图、线上也喂 0–255，分布一致。
  如果像旧版那样在 C++ 里先 `/255`，校准和线上会差 255 倍，INT8 直接崩。
- 300 帧比 120 帧在实拍素材上把中心误差从 1.66px 降到 1.57px，选段也回到 8–12s（合成素材）。

### 结果

| 指标 | FP16 | INT8 |
|---|---|---|
| 检测 p50 / p95 | 91.7 / 130.5 ms | **46.4 / 57.1 ms** |
| 模型大小 | 7.6 MB | **4.3 MB** |
| 进程 RSS | 188 MB | **169 MB** |
| 实拍 person recall（对 FP16，IoU 0.5） | — | **97.9%** |
| 实拍全类 recall / precision | — | 95.9% / 99.3% |
| 平均 IoU / score MAE / 中心误差 | — | 0.981 / 0.024 / 1.57 px |
| 合成片段 recall / 平均 IoU | — | 100% / 0.965 |

### 但选段会变

INT8 的检测框和 FP16 逐帧匹配得上，但框有 1–5px 的抖动。跟踪器吃的是框和 score，
合成片段上 protagonist 的 track id 从 2 变成 3，实拍素材从 13 变成 28，最终选段：

| 素材 | FP16 | INT8 |
|---|---|---|
| 12s 三镜头 | 8.0–12.0s | 8.0–12.0s（300 帧校准后恢复） |
| 16s 实拍 | 5.0–12.0s | 4.0–11.0s |

所以仓库默认仍然用 FP16：选段是产品指标，不能只看检测 recall。INT8 模型和脚本保留，
作为"检测层加速"的可选项，也在提示下一步该做的是**跟踪器对量化抖动的鲁棒性**，
而不是继续压检测延迟。

## 试过但没采用的

- **MMSE 校准**：300 帧跑了 20 分钟没结束（RSS 到 13GB），本轮用 normal。不是算法不行，是时间不够。
- **输入也绑 io_mem 但不加 pass_through**：见上，NPU 把 normalize 吃进去，净收益很小。
- **INT8 + 零拷贝**：INT8 模型输入/输出都是 INT8，输出要按 scale/zp 反量化，当前没实现；
  legacy 路径下输入已经是 uint8，只需要 0.4ms，INT8 的瓶颈不在拷贝。

## 复现

```bash
# x86：校准 + 转换
python scripts/make_calib.py --video 素材.mp4 --out data/calib --num 300
python scripts/convert_yolo_rknn.py --onnx yolov8n.onnx --out models/yolov8n.rknn
python scripts/convert_yolo_rknn.py --onnx yolov8n.onnx --quantize \
  --dataset data/calib/dataset.txt --out models/yolov8n_int8.rknn

# 板端
./build/auto_editor --input ... --model ../models/yolov8n.rknn --bench 200 \
  --bench-json ../assets/bench_fp16_zerocopy.json
./build/auto_editor --input ... --model ../models/yolov8n_int8.rknn --bench 200 \
  --bench-json ../assets/bench_int8.json
./build/auto_editor --input ... --model ../models/yolov8n.rknn \
  --compare-model ../models/yolov8n_int8.rknn --bench 100 --no-zero-copy
```
