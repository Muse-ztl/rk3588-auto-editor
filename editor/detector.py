"""YOLOv8 detector: letterbox -> backend -> DFL decode -> NMS.

同时支持 RKNN（板端 NPU）和 ONNX Runtime（x86 调试）。输出统一为
(x1, y1, x2, y2, score, class_id) 的 float32 数组。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .backends import BaseBackend, letterbox

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int

    @property
    def xyxy(self):
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self):
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def label(self) -> str:
        return COCO80[self.cls] if 0 <= self.cls < len(COCO80) else f"cls{self.cls}"


class YoloV8Detector:
    def __init__(
        self,
        backend: BaseBackend,
        imgsz: int = 640,
        conf: float = 0.30,
        iou: float = 0.45,
        max_det: int = 100,
    ) -> None:
        self.backend = backend
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self._timing_ms = 0.0

    # ------------------------------------------------------------------ pre
    def preprocess(self, bgr: np.ndarray):
        img, ratio, pad = letterbox(bgr, self.imgsz)
        hwc = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, HWC
        return self.backend.adapt(hwc), ratio, pad

    # ---------------------------------------------------------------- post
    def _decode(self, out: np.ndarray):
        """标准单输出 (1, 4+nc, N) 或 (1, N, 4+nc) -> boxes/conf。"""
        pred = np.squeeze(out, 0)
        if pred.ndim != 2:
            raise ValueError(f"unexpected yolo output shape: {out.shape}")
        # 通道在前时形状为 (84, 8400)，转置成 (N, 4+nc)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        cls_ids = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(cls_ids)), cls_ids]
        return boxes_xywh, scores, cls_ids

    # -------------------------------------------------- DFL 多尺度输出解析
    @staticmethod
    def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
        m = x.max(axis=axis, keepdims=True)
        e = np.exp(x - m)
        return e / (e.sum(axis=axis, keepdims=True) + 1e-9)

    def _dfl(self, box: np.ndarray) -> np.ndarray:
        """(1, 64, H, W) 分布 -> (1, 4, H, W) 的 ltrb 距离（网格单位）。"""
        n, c, h, w = box.shape
        bins = c // 4
        y = box.reshape(n, 4, bins, h, w)
        y = self._softmax(y, axis=2)
        acc = np.zeros((n, 4, h, w), dtype=np.float32)
        for i in range(bins):
            acc += y[:, :, i, :, :] * i
        return acc

    def _decode_multiscale(self, outs: List[np.ndarray]):
        """RKNN/ONNX 导出的 YOLOv8 三尺度输出（每个尺度 box/cls[/sum]）。

        box: (1, 64, H, W) DFL 分布；cls: (1, nc, H, W) 已过 sigmoid。
        """
        stride = self.imgsz
        all_boxes, all_scores, all_cls = [], [], []
        for i in range(0, len(outs), 3):
            box = np.asarray(outs[i], dtype=np.float32)
            cls = np.asarray(outs[i + 1], dtype=np.float32)
            _, _, gh, gw = box.shape
            stride = self.imgsz // gh
            dist = self._dfl(box)[0]                      # (4, gh, gw)
            col, row = np.meshgrid(np.arange(gw), np.arange(gh))
            cx = (col + 0.5) * stride
            cy = (row + 0.5) * stride
            x1 = cx - dist[0] * stride
            y1 = cy - dist[1] * stride
            x2 = cx + dist[2] * stride
            y2 = cy + dist[3] * stride
            xyxy = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)

            scores = cls[0].reshape(cls.shape[1], -1).T   # (H*W, nc)
            ids = scores.argmax(1)
            best = scores[np.arange(len(ids)), ids]
            all_boxes.append(xyxy)
            all_scores.append(best)
            all_cls.append(ids)
        return (
            np.concatenate(all_boxes),
            np.concatenate(all_scores),
            np.concatenate(all_cls),
        )

    def _nms(self, boxes_xywh: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray,
             ratio: float, pad, already_xyxy: bool = False) -> List[Box]:
        import cv2

        if already_xyxy:
            xyxy = boxes_xywh
        else:
            xyxy = np.empty_like(boxes_xywh)
            xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
            xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
            xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
            xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        keep: List[Box] = []
        for c in np.unique(cls_ids):
            m = cls_ids == c
            idx = np.where(m)[0]
            xywh = np.stack([
                xyxy[idx, 0], xyxy[idx, 1],
                xyxy[idx, 2] - xyxy[idx, 0], xyxy[idx, 3] - xyxy[idx, 1],
            ], axis=1)
            kept = cv2.dnn.NMSBoxes(xywh.tolist(), scores[idx].tolist(), self.conf, self.iou)
            if len(kept) == 0:
                continue
            for k in np.asarray(kept).reshape(-1):
                i = idx[k]
                x1 = (xyxy[i, 0] - pad[0]) / ratio
                y1 = (xyxy[i, 1] - pad[1]) / ratio
                x2 = (xyxy[i, 2] - pad[0]) / ratio
                y2 = (xyxy[i, 3] - pad[1]) / ratio
                keep.append(Box(float(x1), float(y1), float(x2), float(y2),
                                float(scores[i]), int(c)))
        keep.sort(key=lambda b: -b.score)
        return keep[: self.max_det]

    # ------------------------------------------------------------------ api
    def detect(self, bgr: np.ndarray) -> List[Box]:
        import time

        t0 = time.perf_counter()
        tensor, ratio, pad = self.preprocess(bgr)
        outs = self.backend.infer(tensor)

        if len(outs) >= 6 and outs[0].ndim == 4:
            boxes_xyxy, scores, cls_ids = self._decode_multiscale(outs)
            keep = (scores >= self.conf) & (cls_ids < len(COCO80))
            result = self._nms(boxes_xyxy[keep], scores[keep], cls_ids[keep], ratio, pad,
                               already_xyxy=True)
        else:
            boxes_xywh, scores, cls_ids = self._decode(outs[0])
            keep = (scores >= self.conf) & (cls_ids < len(COCO80))
            result = self._nms(boxes_xywh[keep], scores[keep], cls_ids[keep], ratio, pad)

        self._timing_ms = (time.perf_counter() - t0) * 1e3
        return result

    @property
    def timing_ms(self) -> float:
        return self._timing_ms

    def close(self) -> None:
        self.backend.close()
