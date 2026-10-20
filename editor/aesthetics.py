"""美学/质量打分：决定"这一段值不值得剪进成片"。

全部是可在板端 CPU 上跑的传统指标，单帧 ~2ms：
  * 清晰度：Laplacian 方差（对焦/运动模糊）
  * 色彩：Hasler-Süsstrunk colorfulness
  * 曝光：亮度直方图两端裁剪比例
  * 对比度：灰度标准差
  * 构图：主体中心与三分线的偏离 + 主体画面占比
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from .detector import Box


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _colorfulness(bgr: np.ndarray) -> float:
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _exposure_penalty(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum() + 1e-9
    dark = hist[:12].sum() / total
    bright = hist[-12:].sum() / total
    return float(min(1.0, dark * 2.0 + bright * 2.0))


def _composition(subject: Optional[Box], w: int, h: int) -> float:
    if subject is None:
        return 0.35
    cx, cy = subject.center
    nx, ny = cx / max(w, 1), cy / max(h, 1)
    # 三分线交点距离（0~1 归一化后取最近的一点）
    targets = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
    d = min(((nx - tx) ** 2 + (ny - ty) ** 2) ** 0.5 for tx, ty in targets)
    rule = max(0.0, 1.0 - d / 0.6)
    area = subject.area / max(1.0, w * h)
    # 主体占比 8%~45% 之间给高分，太小看不清、太大没有呼吸感
    if area < 0.08:
        size_score = area / 0.08
    elif area > 0.45:
        size_score = max(0.0, 1.0 - (area - 0.45) / 0.4)
    else:
        size_score = 1.0
    return float(0.55 * rule + 0.45 * size_score)


def score_frame(
    frame: np.ndarray, subject: Optional[Box], ref_sharpness: float = 200.0
) -> Dict[str, float]:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    sharp = _sharpness(gray)
    color = _colorfulness(small)
    exposure = _exposure_penalty(gray)
    contrast = float(gray.std()) / 64.0

    sharp_n = min(1.0, sharp / max(ref_sharpness, 1e-6))
    color_n = min(1.0, color / 60.0)
    contrast_n = min(1.0, contrast)
    comp = _composition(subject, w, h)

    score = (
        0.32 * sharp_n
        + 0.20 * color_n
        + 0.14 * contrast_n
        + 0.20 * comp
        + 0.14 * (1.0 - exposure)
    )
    return {
        "sharpness": sharp,
        "sharpness_n": sharp_n,
        "colorfulness": color,
        "colorfulness_n": color_n,
        "contrast_n": contrast_n,
        "composition": comp,
        "exposure_penalty": exposure,
        "score": float(np.clip(score, 0.0, 1.0)),
    }
