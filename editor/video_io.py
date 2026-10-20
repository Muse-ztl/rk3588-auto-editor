"""视频 I/O：rkmpi 硬解 + 硬编（板端）/ libx264（回退）。

板端把解码和编码都交给 VPU（h264_rkmpp），CPU 只做模型前后处理——
这是简历里"编解码卸载至 VPU"的实际落点。x86 上没有 rkmpp，自动回退。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def has_encoder(name: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    out = _run(["ffmpeg", "-hide_banner", "-encoders"]).stdout.decode("utf-8", "ignore")
    return name in out


def has_decoder(name: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    out = _run(["ffmpeg", "-hide_banner", "-decoders"]).stdout.decode("utf-8", "ignore")
    return name in out


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    n_frames: int
    codec: str

    @property
    def frame_size(self):
        return self.width, self.height


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = n / fps if fps > 0 else 0.0
    codec = ""
    if shutil.which("ffprobe"):
        r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                  "-show_entries", "stream=codec_name", "-of", "json", path])
        try:
            codec = json.loads(r.stdout.decode())["streams"][0]["codec_name"]
        except Exception:
            codec = ""
    return VideoInfo(path, w, h, fps, duration, n, codec)


class FrameReader:
    """默认用 OpenCV(FFmpeg) 读帧；板端可切到 rkmpp 硬解管道。"""

    def __init__(self, path: str, use_rkmpp: bool = False) -> None:
        self.path = path
        self.use_rkmpp = bool(use_rkmpp and has_decoder("h264_rkmpp"))
        self.info = probe(path)
        self._proc: Optional[subprocess.Popen] = None
        self._cap: Optional[cv2.VideoCapture] = None
        if self.use_rkmpp:
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-c:v", "h264_rkmpp",
                "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"
            ]
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        else:
            self._cap = cv2.VideoCapture(path)

    @property
    def backend(self) -> str:
        return "ffmpeg-rkmpp" if self.use_rkmpp else "opencv-ffmpeg"

    def read(self) -> Optional[np.ndarray]:
        w, h = self.info.frame_size
        if self._proc is not None:
            raw = self._proc.stdout.read(w * h * 3)
            if not raw or len(raw) < w * h * 3:
                return None
            return np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._proc is not None:
            self._proc.stdout.close()
            self._proc.terminate()
            self._proc = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class VideoWriter:
    def __init__(self, out_path: str, info: VideoInfo, prefer_rkmpp: bool = True) -> None:
        self.out_path = out_path
        self.info = info
        if prefer_rkmpp and has_encoder("h264_rkmpp"):
            self.venc = ["-c:v", "h264_rkmpp", "-b:v", "8M"]
            self.encoder = "h264_rkmpp"
        else:
            self.venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22"]
            self.encoder = "libx264"

    def cut(self, start: float, duration: float, out_path: Optional[str] = None) -> str:
        """精确切割一个片段并统一编码参数，方便后续无损 concat。"""
        out = out_path or self.out_path
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", self.info.path,
            "-an", *self.venc, "-pix_fmt", "yuv420p", "-r", f"{self.info.fps:.3f}", out,
        ]
        r = _run(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg cut failed: {r.stderr.decode('utf-8', 'ignore')[-400:]}")
        return out

    def concat(self, parts: List[str], out_path: Optional[str] = None) -> str:
        out = out_path or self.out_path
        fd, list_path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(list_path, "w", encoding="utf-8") as f:
            for p in parts:
                f.write(f"file '{os.path.abspath(p)}'\n")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out,
        ]
        r = _run(cmd)
        os.unlink(list_path)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {r.stderr.decode('utf-8', 'ignore')[-400:]}")
        return out
