"""命令行入口：端侧视频自动剪辑。

例子（板端）：
    python scripts/run_edit.py \
        --input data/demo_3scenes.mp4 \
        --output output/final_cut.mp4 \
        --backend rknn --model models/yolov8n.rknn \
        --analysis-fps 12 --report output/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from editor.pipeline import run  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="On-device auto video editor (RK3588S)")
    p.add_argument("--input", required=True, help="source video")
    p.add_argument("--output", default="output/final_cut.mp4")
    p.add_argument("--backend", default="rknn", choices=["rknn", "onnxruntime", "ort"])
    p.add_argument("--model", default="models/yolov8n.rknn",
                   help=".rknn for npu backend, .onnx for ort backend")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--analysis-fps", type=float, default=12.0,
                   help="detection sampling rate (fps of the analysis pass)")
    p.add_argument("--no-rkmpp", action="store_true", help="disable VPU decode/encode")
    p.add_argument("--report", default=None)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    report = run(
        path=a.input,
        out_path=a.output,
        backend=a.backend,
        model_path=a.model,
        conf=a.conf,
        analysis_fps=a.analysis_fps,
        use_rkmpp=not a.no_rkmpp,
        report_path=a.report,
    )
    print("\n=== edit report ===")
    print(f"video          : {report.video.get('duration', 0):.1f}s "
          f"{report.video.get('width')}x{report.video.get('height')} @{report.video.get('fps', 0):.1f}fps")
    print(f"decode/encode  : {report.decode_backend} / {report.encode_backend}")
    print(f"detector       : {report.detector_backend}  p50={report.detect_ms_p50:.1f}ms "
          f"p90={report.detect_ms_p90:.1f}ms")
    print(f"analysis       : {report.analyzed_frames} frames @ {report.analysis_fps:.1f}fps, "
          f"{report.pass1_seconds:.1f}s (detect {report.detect_calls}, reid {report.reid_calls}, "
          f"aesthetic {report.aesthetic_calls})")
    if report.protagonist:
        print(f"protagonist    : {report.protagonist}")
    print(f"segments       : {[(round(s['start'], 1), round(s['end'], 1), round(s['score'], 2)) for s in report.segments]}")
    print(f"output         : {report.output_duration:.1f}s in {report.pass2_seconds:.1f}s "
          f"(render speedup {report.video.get('duration', 0) / max(report.pass2_seconds, 1e-6):.1f}x)")
    if a.report:
        print(f"report json    : {a.report}")


if __name__ == "__main__":
    main()
