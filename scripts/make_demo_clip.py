"""生成测试素材：同一张图做三档"变焦"，模拟推镜。

真实素材涉及隐私不便公开，这里用一张公开样例图合成可复现的素材：
    scene A  1.0×（全景，人很小）
    scene B  1.6×（中景）
    scene C  2.4×（近景，人占画面主体，且缓慢横移）

自动剪辑应当优先选 C 段：人物显著、画面干净。三档变焦都保持 16:9，
不会像硬裁剪那样把人切出画面（那样检测直接失效，是早期版本踩过的坑）。
"""

from __future__ import annotations

import argparse
import os
import subprocess

import cv2
import numpy as np


def _writer(path: str, w: int, h: int, fps: int) -> subprocess.Popen:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/sample.jpg")
    ap.add_argument("--out", default="data/demo_3scenes.mp4")
    ap.add_argument("--seconds", type=float, default=4.0, help="每段时长")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--focus", default="0.35,0.70", help="关注点（归一化 x,y），一般是人物位置")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    fx, fy = (float(v) for v in args.focus.split(","))
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    ih, iw = img.shape[:2]

    zooms = [1.0, 1.6, 2.4]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    tmp_parts = []
    for zi, z in enumerate(zooms):
        part = f"{args.out}.part{zi}.mp4"
        proc = _writer(part, w, h, args.fps)
        n = int(args.seconds * args.fps)
        for i in range(n):
            t = i / max(n - 1, 1)
            # 近景给一点横移，避免完全静止
            drift = (t - 0.5) * (0.06 / z)
            cx = np.clip(fx + drift, 0.0, 1.0) * iw
            cy = np.clip(fy, 0.0, 1.0) * ih
            cw = iw / z
            ch = min(ih, cw * h / w)
            cw = ch * w / h
            x1 = int(np.clip(cx - cw / 2, 0, max(0, iw - cw)))
            y1 = int(np.clip(cy - ch / 2, 0, max(0, ih - ch)))
            crop = img[y1:y1 + int(ch), x1:x1 + int(cw)]
            frame = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
        tmp_parts.append(part)

    list_file = f"{args.out}.list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in tmp_parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", args.out],
                   check=True)
    for p in tmp_parts:
        os.unlink(p)
    os.unlink(list_file)
    print(f"wrote {args.out} ({args.seconds * len(zooms):.0f}s, {w}x{h}@{args.fps})")


if __name__ == "__main__":
    main()
