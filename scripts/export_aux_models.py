"""导出/下载剪辑流水线用到的三个辅助模型（x86 上跑，产物放 models/）。

    python scripts/export_aux_models.py --out models

三个模型都是可选的：运行时缺哪个，对应模块自动关闭（镜头切分退化为直方图）。

  yunet.onnx          人脸检测（OpenCV Zoo，固定 640×640 输入）
  yolov8n-pose.onnx   姿态（ultralytics 导出，1×3×640×640 -> 1×56×8400）
  transnetv2.onnx     镜头切分（transnetv2-pytorch 权重导出，1×100×27×48×3 uint8）
"""

from __future__ import annotations

import argparse
import os
import urllib.request

YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def fetch_yunet(out_dir: str) -> None:
    dst = os.path.join(out_dir, "yunet.onnx")
    if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
        print(f"[skip] {dst}")
        return
    print(f"[get ] {YUNET_URL}")
    urllib.request.urlretrieve(YUNET_URL, dst)
    print(f"       -> {dst} ({os.path.getsize(dst) / 1e3:.0f} KB)")


def export_pose(out_dir: str) -> None:
    dst = os.path.join(out_dir, "yolov8n-pose.onnx")
    if os.path.exists(dst):
        print(f"[skip] {dst}")
        return
    from ultralytics import YOLO

    model = YOLO("yolov8n-pose.pt")  # 首次运行会从 ultralytics assets 下载
    path = model.export(format="onnx", opset=12, simplify=True, imgsz=640, dynamic=False)
    os.replace(path, dst)
    print(f"       -> {dst}")


def export_transnet(out_dir: str) -> None:
    dst = os.path.join(out_dir, "transnetv2.onnx")
    if os.path.exists(dst):
        print(f"[skip] {dst}")
        return
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device="cpu").eval()
    dummy = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)
    os.makedirs(out_dir, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), dst, opset_version=17, dynamo=False,
        input_names=["frames"], output_names=["logits"],
    )
    print(f"       -> {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    ap.add_argument("--only", default="", help="yunet / pose / transnet")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    todo = args.only.split(",") if args.only else ["yunet", "pose", "transnet"]
    if "yunet" in todo:
        fetch_yunet(args.out)
    if "pose" in todo:
        export_pose(args.out)
    if "transnet" in todo:
        export_transnet(args.out)


if __name__ == "__main__":
    main()
