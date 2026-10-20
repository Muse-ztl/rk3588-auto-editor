"""主角识别 + 片段打分 + 剪辑决策（EDL）。

主角判定不是"出现最久的人"，而是多信号加权：曝光时长、画面中心度、清晰度、
是否正脸（有人脸/姿态模块时）、检测置信度。这样能避开"背景里路过的人"和
"镜头边缘只露半张脸"的情况。

两阶段打分（关键）：
  pass1 只能记录"每个 1s 桶里各 track 出现了几帧"，因为主角要等整段分析完
  才能确定；确定主角后再回头算每个桶的主角占比，避免边分析边猜主角导致的抖动。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrackStat:
    tid: int
    cls: int
    frames: int = 0
    score_sum: float = 0.0
    center_sum: float = 0.0
    sharp_sum: float = 0.0
    face_hits: int = 0

    @property
    def mean_score(self) -> float:
        return self.score_sum / max(self.frames, 1)

    @property
    def mean_center(self) -> float:
        return self.center_sum / max(self.frames, 1)

    @property
    def mean_sharp(self) -> float:
        return self.sharp_sum / max(self.frames, 1)

    def protagonist_score(self) -> float:
        return (
            0.34 * min(1.0, self.frames / 60.0)   # 曝光时长（饱和到 60 帧）
            + 0.28 * self.mean_center             # 越靠画面中心越可能是主角
            + 0.20 * self.mean_sharp              # 主体是否在焦内
            + 0.10 * min(1.0, self.face_hits / 10.0)
            + 0.08 * self.mean_score
        )


@dataclass
class Segment:
    start: float
    end: float
    score: float = 0.0
    protagonist_ratio: float = 0.0
    aesthetic: float = 0.0
    note: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class HighlightPicker:
    def __init__(
        self,
        fps: float,
        segment_seconds: float = 1.0,
        min_clip: float = 3.0,
        max_clip: float = 8.0,
        target_ratio: float = 0.45,
        merge_gap: float = 1.2,
    ) -> None:
        self.fps = max(fps, 1e-3)
        self.segment_seconds = segment_seconds
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.target_ratio = target_ratio
        self.merge_gap = merge_gap

        self.stats: Dict[int, TrackStat] = {}
        self.protag_tid: Optional[int] = None
        self._frames: Dict[int, int] = {}
        # 每桶每 track 的"显著性"累加（salience = 中心度 × 主体大小），
        # 只看帧数会让"远处路过的小人"和"近景特写"等价，权重必须带尺寸。
        self._tid_sal: Dict[int, Dict[int, float]] = {}
        self._aes: Dict[int, Tuple[float, int]] = {}
        # 硬切场景里 track id 会重建，按 id 统计主角占比不稳；
        # 片段评分改用"该秒内人物的最大显著性"，跨 ID 也成立。
        self._bucket_prom: Dict[int, float] = {}

    # ----------------------------------------------------------- accumulate
    def add_frame(
        self,
        t: float,
        tracks,
        track_sharp: Dict[int, float],
        center_score: Dict[int, float],
        aesthetic: Optional[float] = None,
    ) -> None:
        bucket = int(t // self.segment_seconds)
        self._frames[bucket] = self._frames.get(bucket, 0) + 1
        per_tid = self._tid_sal.setdefault(bucket, {})

        for tr in tracks:
            per_tid[tr.tid] = per_tid.get(tr.tid, 0.0) + float(center_score.get(tr.tid, 0.0))
            st = self.stats.setdefault(tr.tid, TrackStat(tid=tr.tid, cls=tr.cls))
            st.frames += 1
            st.score_sum += float(tr.score)
            st.center_sum += float(center_score.get(tr.tid, 0.0))
            st.sharp_sum += float(track_sharp.get(tr.tid, 0.0))

        prom = max((float(center_score.get(tr.tid, 0.0)) for tr in tracks if tr.cls == 0),
                   default=0.0)
        self._bucket_prom[bucket] = max(self._bucket_prom.get(bucket, 0.0), prom)

        if aesthetic is not None:
            s, n = self._aes.get(bucket, (0.0, 0))
            self._aes[bucket] = (s + aesthetic, n + 1)

    # ---------------------------------------------------------------- pick
    def choose_protagonist(self) -> Optional[TrackStat]:
        if not self.stats:
            return None
        persons = [s for s in self.stats.values() if s.cls == 0 and s.frames >= 3]
        pool = persons or [s for s in self.stats.values() if s.frames >= 3]
        if not pool:
            return None
        best = max(pool, key=lambda s: s.protagonist_score())
        self.protag_tid = best.tid
        return best

    def build_segments(self, total_duration: float) -> List[Segment]:
        if not self._frames:
            return []

        buckets = sorted(self._frames)
        frame_counts = [self._frames[b] for b in buckets]

        # 美学是稀疏采样（1fps），缺桶用前一个值前向填充，避免出现"空洞"把片段切碎
        aes_vals: List[float] = []
        last = 0.0
        for b in buckets:
            s, n = self._aes.get(b, (0.0, 0))
            if n:
                last = s / n
            aes_vals.append(last)

        # 显著性经验上落在 0~0.9，除以 0.75 归一到 0~1
        ratios = [min(1.0, self._bucket_prom.get(b, 0.0) / 0.75) for b in buckets]

        raw = [0.62 * r + 0.38 * a for r, a in zip(ratios, aes_vals)]
        # 3 桶滑动平均，抑制单帧抖动
        smooth = []
        for i in range(len(raw)):
            lo, hi = max(0, i - 1), min(len(raw), i + 2)
            smooth.append(statistics.mean(raw[lo:hi]))

        # 滑动窗口选段：比"阈值分组"稳。阈值分组在全片都不错时会退化成
        # 一整段，然后被 max_clip 从 0s 砍断——那正好丢掉最精彩的后半段。
        win_len = max(self.min_clip, min(self.max_clip, total_duration * self.target_ratio))
        win = max(1, int(round(win_len / self.segment_seconds)))

        candidates: List[Segment] = []
        for i in range(max(1, len(buckets) - win + 1)):
            ws = smooth[i:i + win]
            if not ws:
                continue
            start_t = buckets[i] * self.segment_seconds
            end_t = min(total_duration, start_t + win * self.segment_seconds)
            candidates.append(
                Segment(
                    start=start_t,
                    end=end_t,
                    score=statistics.mean(ws),
                    protagonist_ratio=statistics.mean(ratios[i:i + win]),
                    aesthetic=statistics.mean(aes_vals[i:i + win]),
                )
            )

        if not candidates:
            return []

        target = max(self.min_clip, total_duration * self.target_ratio)
        chosen: List[Segment] = []
        budget = target * 1.15
        for seg in sorted(candidates, key=lambda c: -c.score):
            used = sum(c.duration for c in chosen)
            if used >= target:
                break
            if used + seg.duration > budget:
                continue
            if any(seg.start < c.end and c.start < seg.end for c in chosen):
                continue
            chosen.append(seg)

        if not chosen:
            chosen.append(max(candidates, key=lambda c: c.score))

        chosen.sort(key=lambda s: s.start)
        return chosen
