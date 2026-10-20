"""外观特征（ReID）。

板端默认走 HSV 直方图：零依赖、单帧 <1ms，对"同一段素材里的主角追踪"足够用。
如果提供 OSNet 的 RKNN/ONNX 权重，则切换成 512 维深度特征（跨镜头更稳）。
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .backends import BaseBackend
from .detector import Box


class AppearanceEncoder:
    def __init__(
        self,
        backend: Optional[BaseBackend] = None,
        input_hw: tuple[int, int] = (256, 128),
    ) -> None:
        self.backend = backend
        self.input_hw = input_hw

    # ---------------------------------------------------------------- hist
    @staticmethod
    def _hsv_feat(crop: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        feat = np.concatenate([h, s, v]).astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9)

    # -------------------------------------------------------------- osnet
    def _osnet_feat(self, crops: List[np.ndarray]) -> np.ndarray:
        batch = np.zeros((len(crops), 3, *self.input_hw), dtype=np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for i, c in enumerate(crops):
            img = cv2.resize(c, (self.input_hw[1], self.input_hw[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - mean) / std
            batch[i] = img.transpose(2, 0, 1)
        out = self.backend.infer(batch)[0]
        out = np.asarray(out).reshape(len(crops), -1)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)

    # ----------------------------------------------------------------- api
    def encode(self, frame: np.ndarray, boxes: List[Box]) -> np.ndarray:
        if not boxes:
            return np.zeros((0, 32), dtype=np.float32)
        h, w = frame.shape[:2]
        crops = []
        for b in boxes:
            x1 = int(max(0, b.x1)); y1 = int(max(0, b.y1))
            x2 = int(min(w - 1, b.x2)); y2 = int(min(h - 1, b.y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                crops.append(np.zeros((8, 8, 3), dtype=np.uint8))
            else:
                crops.append(frame[y1:y2, x1:x2])

        if self.backend is not None:
            return self._osnet_feat(crops)
        return np.stack([self._hsv_feat(c) for c in crops])
