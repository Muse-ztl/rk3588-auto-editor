"""Convert YOLOv8 ONNX to RKNN (x86 host, rknn-toolkit2 2.3.2).

    python scripts/convert_yolo_rknn.py --onnx yolov8n.onnx --out models/yolov8n.rknn
    python scripts/convert_yolo_rknn.py --onnx yolov8n.onnx --quantize --dataset dataset.txt

说明：
  * 归一化放在 Python 前处理里做（/255 + RGB），所以这里 mean=0 / std=1，
    否则会重复缩放导致输出全错。
  * 默认 FP16。INT8 需要 calibration 数据集（dataset.txt 每行一个图片路径），
    在 RK3588 上单帧可从 ~110ms 降到 ~50ms，但小目标置信度会掉。
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", default="models/yolov8n.rknn")
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--dataset", default="")
    args = ap.parse_args()

    from rknn.api import RKNN

    if args.quantize and not (args.dataset and os.path.exists(args.dataset)):
        raise SystemExit("--quantize requires an existing --dataset list file")

    rknn = RKNN(verbose=False)
    try:
        print(f"[convert] config mean=0 std=1 target={args.target}")
        assert rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                           target_platform=args.target) == 0
        print(f"[convert] load {args.onnx}")
        assert rknn.load_onnx(model=args.onnx) == 0
        kwargs = {"do_quantization": bool(args.quantize)}
        if args.quantize:
            kwargs["dataset"] = args.dataset
        print(f"[convert] build quantize={args.quantize}")
        assert rknn.build(**kwargs) == 0
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        assert rknn.export_rknn(args.out) == 0
        print(f"[convert] -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
