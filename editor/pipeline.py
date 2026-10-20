"""端侧视频自动剪辑流水线（分层采样调度）。

一次分析 pass + 一次剪辑 pass：

  pass1（分析）：
      解码 -> 按信号频率分层采样
        · 检测 + 跟踪：每帧（12fps 分析率）
        · 外观特征：每 2 帧（给跟踪器做 ID 保持）
        · 美学打分：每 12 帧（1fps，质量变化本来就慢）
      -> 主角识别 / 1s 片段打分 -> EDL

  pass2（成片）：
      EDL -> ffmpeg 精确切割 -> 统一编码参数 -> concat 无损拼接

板端 pass1 用 NPU + rkmpp 解码；x86 上自动回退到 ONNX Runtime + OpenCV。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .aesthetics import score_frame
from .detector import Box, YoloV8Detector
from .highlight import HighlightPicker, Segment
from .reid import AppearanceEncoder
from .tracker import BoTSORT
from .video_io import FrameReader, VideoInfo, VideoWriter, probe


@dataclass
class EditorReport:
    input: str
    output: str
    video: dict = field(default_factory=dict)
    decode_backend: str = ""
    encode_backend: str = ""
    detector_backend: str = ""
    analysis_fps: float = 0.0
    analyzed_frames: int = 0
    detect_calls: int = 0
    reid_calls: int = 0
    aesthetic_calls: int = 0
    detect_ms_p50: float = 0.0
    detect_ms_p90: float = 0.0
    pass1_seconds: float = 0.0
    pass2_seconds: float = 0.0
    protagonist: Optional[dict] = None
    segments: List[dict] = field(default_factory=list)
    output_duration: float = 0.0
    speedup_vs_realtime: float = 0.0


def _pct(vals, p):
    return float(np.percentile(vals, p)) if vals else 0.0


def analyze(
    path: str,
    detector: YoloV8Detector,
    tracker: BoTSORT,
    encoder: AppearanceEncoder,
    analysis_fps: Optional[float] = None,
    use_rkmpp: bool = True,
    progress_every: int = 60,
) -> tuple[HighlightPicker, EditorReport, VideoInfo]:
    info = probe(path)
    fps = info.fps or 25.0
    stride = max(1, int(round(fps / (analysis_fps or min(fps, 12.0)))))
    analysis_rate = fps / stride

    picker = HighlightPicker(fps=analysis_rate)
    report = EditorReport(input=path, output="", analysis_fps=analysis_rate)

    detect_ms: List[float] = []
    t0 = time.perf_counter()

    with FrameReader(path, use_rkmpp=use_rkmpp) as reader:
        report.decode_backend = reader.backend
        idx = 0
        while True:
            frame = reader.read()
            if frame is None:
                break
            if idx % stride != 0:
                idx += 1
                continue

            t = idx / fps
            boxes = detector.detect(frame)
            report.detect_calls += 1
            detect_ms.append(detector.timing_ms)

            feats = None
            if idx % (stride * 2) == 0:
                feats = encoder.encode(frame, boxes)
                report.reid_calls += 1

            tracks = tracker.update(boxes, feats, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            center_scores: Dict[int, float] = {}
            sharp_scores: Dict[int, float] = {}
            h, w = frame.shape[:2]
            for tr in tracks:
                x1, y1, x2, y2 = tr.box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                centrality = float(max(0.0, 1.0 - np.hypot(cx - w / 2, cy - h / 2) /
                                       np.hypot(w / 2, h / 2)))
                area_ratio = float(max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, w * h))
                # 显著性 = 中心度 + 主体大小：近景特写比远景路人更该当主角
                center_scores[tr.tid] = 0.35 * centrality + 0.65 * float(np.sqrt(min(1.0, area_ratio / 0.40)))
                crop = frame[int(max(0, y1)):int(min(h, y2)), int(max(0, x1)):int(min(w, x2))]
                if crop.size:
                    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    sharp_scores[tr.tid] = min(1.0, float(cv2.Laplacian(g, cv2.CV_64F).var()) / 300.0)

            aesthetic = None
            protag_box = None
            if idx % (stride * 12) == 0:
                # 美学按 1fps 采样：先选当前得分最高的 person 作为主体
                persons = [tr for tr in tracks if tr.cls == 0] or tracks
                if persons:
                    protag_box = max(persons, key=lambda tr: tr.score).as_box()
                aesthetic = score_frame(frame, protag_box)["score"]
                report.aesthetic_calls += 1

            picker.add_frame(t, tracks, sharp_scores, center_scores, aesthetic)
            report.analyzed_frames += 1

            if report.analyzed_frames % progress_every == 0:
                print(f"[analyze] {t:.1f}s / {info.duration:.1f}s  tracks={len(tracks)}")

            idx += 1

    report.pass1_seconds = time.perf_counter() - t0
    report.detector_backend = detector.backend.name
    report.detect_ms_p50 = _pct(detect_ms, 50)
    report.detect_ms_p90 = _pct(detect_ms, 90)
    report.video = asdict(info)
    protag = picker.choose_protagonist()
    if protag:
        report.protagonist = {
            "track_id": protag.tid,
            "class_id": protag.cls,
            "frames": protag.frames,
            "score": round(protag.protagonist_score(), 4),
            "mean_center": round(protag.mean_center, 4),
        }
    return picker, report, info


def render(
    picker: HighlightPicker,
    report: EditorReport,
    info: VideoInfo,
    out_path: str,
    prefer_rkmpp: bool = True,
) -> EditorReport:
    segments: List[Segment] = picker.build_segments(info.duration)
    report.segments = [asdict(s) for s in segments]

    t0 = time.perf_counter()
    writer = VideoWriter(out_path, info, prefer_rkmpp=prefer_rkmpp)
    report.encode_backend = writer.encoder

    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), ".segments")
    os.makedirs(tmp_dir, exist_ok=True)
    parts: List[str] = []
    for i, seg in enumerate(segments):
        part = os.path.join(tmp_dir, f"seg_{i:02d}.mp4")
        writer.cut(seg.start, seg.duration, part)
        parts.append(part)

    if not parts:
        raise RuntimeError("no segment selected; check detector output")

    writer.concat(parts, out_path)
    for p in parts:
        os.unlink(p)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    report.pass2_seconds = time.perf_counter() - t0
    out_info = probe(out_path)
    report.output = out_path
    report.output_duration = out_info.duration
    total = report.pass1_seconds + report.pass2_seconds
    report.speedup_vs_realtime = (info.duration / total) if total > 0 else 0.0
    return report


def run(
    path: str,
    out_path: str,
    backend: str = "rknn",
    model_path: str = "models/yolov8n.rknn",
    conf: float = 0.30,
    analysis_fps: Optional[float] = None,
    use_rkmpp: bool = True,
    report_path: Optional[str] = None,
) -> EditorReport:
    from .backends import build_backend

    be = build_backend(backend, model_path)
    detector = YoloV8Detector(be, conf=conf)
    tracker = BoTSORT()
    encoder = AppearanceEncoder()

    try:
        picker, report, info = analyze(
            path, detector, tracker, encoder, analysis_fps=analysis_fps, use_rkmpp=use_rkmpp
        )
        report = render(picker, report, info, out_path, prefer_rkmpp=use_rkmpp)
    finally:
        detector.close()

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    return report
