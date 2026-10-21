"""Build an INT8 calibration set for YOLOv8 from videos.

    python scripts/make_calib.py --video 素材.mp4 --out data/calib_yolo --num 120

Frames are letterboxed exactly like the C++ preprocess (114 padding), so the
quantizer sees the same distribution as the runtime. --rgb writes the channels
in RGB order for toolkits that read files with OpenCV's BGR convention.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    x = int(round((size - nw) / 2 - 0.1))
    y = int(round((size - nh) / 2 - 0.1))
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", required=True, help="can be repeated")
    ap.add_argument("--out", default="data/calib_yolo")
    ap.add_argument("--num", type=int, default=120)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--rgb", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_video = max(1, args.num // len(args.video))
    saved = 0
    for video in args.video:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {video}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // per_video)
        for i in range(0, total, step):
            if saved >= args.num:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue
            img = letterbox(frame, args.size)
            if args.rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            name = f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        cap.release()

    files = sorted(out.glob("frame_*.jpg"))
    with open(out / "dataset.txt", "w", encoding="utf-8") as f:
        for p in files:
            f.write(str(p.resolve()) + "\n")
    print(f"[calib] {len(files)} frames -> {out / 'dataset.txt'}")


if __name__ == "__main__":
    main()
