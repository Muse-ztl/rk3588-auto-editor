"""BoT-SORT 跟踪器（Kalman + IoU + 外观特征 + 全局运动补偿）。

为什么不用现成库：板端要控制依赖体积与内存，而且 BoT-SORT 的核心
（8 维常量速度 Kalman + 两级匹配 + GMC）不到 300 行，自己写反而更容易
按需裁剪（例如只在检测框抖动明显时才跑 GMC）。

匹配策略：
  1. 高置信度检测 vs 已确认轨迹：IoU 代价 + 外观余弦代价，Hungarian 求解
  2. 低置信度检测 vs 剩余轨迹：只按 IoU 匹配
  3. 仍未匹配的轨迹按 age 淘汰，未匹配的高分检测新建轨迹
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import Box


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) b: (M,4) xyxy -> (N,M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _assign(cost: np.ndarray) -> List[Tuple[int, int]]:
    """优先用 scipy 的最优匹配；板端没装 scipy 时退化成贪心（代价升序）。

    贪心在目标数 <20 的监控场景里与匈牙利算法结果基本一致，且不需要额外依赖。
    """
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), cols.tolist()))
    except ImportError:
        order = np.argsort(cost, axis=None)
        used_r, used_c, pairs = set(), set(), []
        for flat in order:
            r, c = divmod(int(flat), cost.shape[1])
            if cost[r, c] >= 1e4:
                break
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
        return pairs


class KalmanBox:
    """常量速度模型，状态 [cx, cy, w, h, vcx, vcy, vw, vh]。"""

    def __init__(self, box: Box, std_pos: float = 10.0, std_vel: float = 100.0):
        self._F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._F[i, i + 4] = 1.0
        self._H = np.eye(4, 8, dtype=np.float32)
        self._Q = np.eye(8, dtype=np.float32) * 1e-2
        self._R = np.eye(4, dtype=np.float32) * 1e-1
        self._P = np.eye(8, dtype=np.float32)
        self._P[4:, 4:] *= 1000.0
        self._P *= 10.0
        self.x = self._to_state(box)

    @staticmethod
    def _to_state(box: Box) -> np.ndarray:
        return np.array([
            (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2,
            box.x2 - box.x1, box.y2 - box.y1, 0, 0, 0, 0,
        ], dtype=np.float32)

    def predict(self, shift: Optional[np.ndarray] = None) -> np.ndarray:
        if shift is not None:
            self.x[:2] += shift
        self.x = self._F @ self.x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self.x

    def update(self, box: Box) -> None:
        # 量测只有 4 维 (cx, cy, w, h)，速度分量由预测给出
        z = self._to_state(box)[:4]
        y = z - self._H @ self.x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self._P = (np.eye(8, dtype=np.float32) - K @ self._H) @ self._P

    @property
    def xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


@dataclass
class Track:
    tid: int
    cls: int
    kf: KalmanBox
    score: float
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    confirmed: bool = False
    feature: Optional[np.ndarray] = None
    history: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def box(self) -> np.ndarray:
        return self.kf.xyxy

    def as_box(self) -> Box:
        x1, y1, x2, y2 = self.box
        return Box(float(x1), float(y1), float(x2), float(y2), float(self.score), int(self.cls))


class BoTSORT:
    def __init__(
        self,
        iou_gate: float = 0.3,
        appearance_weight: float = 0.35,
        max_age: int = 30,
        min_hits: int = 2,
        use_gmc: bool = True,
    ) -> None:
        self.iou_gate = iou_gate
        self.appearance_weight = appearance_weight
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_gmc = use_gmc
        self.tracks: List[Track] = []
        self._next_id = 1
        self._prev_gray: Optional[np.ndarray] = None
        self._gmc_shift = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------ gmc
    def _estimate_gmc(self, gray: np.ndarray) -> np.ndarray:
        import cv2

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return np.zeros(2, dtype=np.float32)
        try:
            orb = cv2.ORB_create(nfeatures=200)
            k1, d1 = orb.detectAndCompute(self._prev_gray, None)
            k2, d2 = orb.detectAndCompute(gray, None)
            if d1 is not None and d2 is not None and len(k1) > 10 and len(k2) > 10:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(d1, d2)
                if len(matches) > 12:
                    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
                    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
                    mat, _ = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC)
                    if mat is not None:
                        self._gmc_shift = np.array([mat[0, 2], mat[1, 2]], dtype=np.float32)
        except Exception:
            pass
        self._prev_gray = gray
        return self._gmc_shift

    # ---------------------------------------------------------------- match
    def _appearance_cost(self, det_feats: np.ndarray, tracks: List[Track]) -> np.ndarray:
        cost = np.ones((len(tracks), len(det_feats)), dtype=np.float32)
        for i, t in enumerate(tracks):
            if t.feature is None:
                continue
            sim = det_feats @ t.feature  # 已 L2 归一化 -> 余弦相似度
            cost[i] = 1.0 - np.clip(sim, -1.0, 1.0)
        return cost

    def update(self, dets: List[Box], feats: Optional[np.ndarray], gray: Optional[np.ndarray]):
        """dets: 当前帧检测框；feats: (N, D) L2 归一化外观特征（可为空）。"""
        shift = self._estimate_gmc(gray) if (self.use_gmc and gray is not None) else None
        for t in self.tracks:
            t.kf.predict(shift)
            t.age += 1
            t.time_since_update += 1

        if not dets:
            self._prune()
            return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

        det_boxes = np.stack([d.xyxy for d in dets])
        trk_boxes = np.stack([t.box for t in self.tracks]) if self.tracks else np.zeros((0, 4))
        iou = _iou_matrix(trk_boxes, det_boxes)

        appearance = None
        if feats is not None and len(self.tracks) > 0:
            appearance = self._appearance_cost(feats, self.tracks)

        # 代价矩阵：IoU 距离为主，外观为辅；IoU 低于门限直接判为不可行
        cost = 1.0 - iou
        if appearance is not None:
            cost = (1.0 - self.appearance_weight) * cost + self.appearance_weight * appearance
        cost[iou < self.iou_gate] = 1e5

        matched: List[Tuple[int, int]] = []
        if self.tracks:
            for r, c in _assign(cost):
                if cost[r, c] < 1e4:
                    matched.append((r, c))

        used_det = {c for _, c in matched}
        used_trk = {r for r, _ in matched}

        for r, c in matched:
            t = self.tracks[r]
            t.kf.update(dets[c])
            t.score = dets[c].score
            t.hits += 1
            t.time_since_update = 0
            t.history.append(dets[c].center)
            if feats is not None:
                t.feature = feats[c] if t.feature is None else 0.9 * t.feature + 0.1 * feats[c]
                t.feature /= (np.linalg.norm(t.feature) + 1e-9)
            if t.hits >= self.min_hits:
                t.confirmed = True

        # 二级匹配：低分检测兜底（这里没有分数分级，按 IoU 再捞一次未被匹配的）
        for i in range(len(self.tracks)):
            if i in used_trk:
                continue
            best, best_iou = -1, self.iou_gate
            for j in range(len(dets)):
                if j in used_det:
                    continue
                v = iou[i, j]
                if v > best_iou:
                    best, best_iou = j, v
            if best >= 0:
                used_det.add(best)
                t = self.tracks[i]
                t.kf.update(dets[best])
                t.score = dets[best].score
                t.hits += 1
                t.time_since_update = 0
                if feats is not None:
                    t.feature = feats[best]
                if t.hits >= self.min_hits:
                    t.confirmed = True

        for j, d in enumerate(dets):
            if j in used_det:
                continue
            if d.score < 0.25:
                continue
            t = Track(
                tid=self._next_id, cls=d.cls, kf=KalmanBox(d), score=d.score,
                feature=None if feats is None else feats[j].copy(),
            )
            t.history.append(d.center)
            self._next_id += 1
            self.tracks.append(t)

        self._prune()
        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

    def _prune(self) -> None:
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]
