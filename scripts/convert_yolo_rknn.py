"""Convert YOLOv8 ONNX to RKNN (x86 host, rknn-toolkit2 2.3.2).

    python scripts/convert_yolo_rknn.py --onnx yolov8n.onnx --out models/yolov8n.rknn
    python scripts/convert_yolo_rknn.py --onnx yolov8n.onnx --quantize --dataset dataset.txt

说明：
  * 归一化放在 RKNN 的 std_values 里做（std=255），C++ 前处理只做 letterbox + RGB，
    直接喂 0~255。这样 INT8 校准时读的原始图片和线上推理是同一分布。
  * 默认 FP16。INT8 需要 calibration 数据集（dataset.txt 每行一个图片路径）。
    本机实测 16s 1080p 素材：检测 p50 91.7ms -> 46.4ms，模型 7.6MB -> 4.3MB，
    person recall 97.9%，但检测框抖动会影响下游跟踪/选段，默认不启用。
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
    ap.add_argument("--quant-algorithm", default="normal",
                    choices=["normal", "mmse", "kl_divergence"])
    ap.add_argument("--hybrid-level", type=int, default=0,
                    help="0 = all INT8; >0 lets sensitive layers fall back to FP16")
    args = ap.parse_args()

    from rknn.api import RKNN

    if args.quantize and not (args.dataset and os.path.exists(args.dataset)):
        raise SystemExit("--quantize requires an existing --dataset list file")

    rknn = RKNN(verbose=False)
    try:
        print(f"[convert] config mean=0 std=255 target={args.target} "
              f"alg={args.quant_algorithm} hybrid={args.hybrid_level}")
        assert rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                           target_platform=args.target,
                           quantized_algorithm=args.quant_algorithm,
                           quantized_hybrid_level=args.hybrid_level) == 0
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
