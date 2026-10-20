"""Inference backends for the editor: RKNN (NPU) with an ONNX Runtime fallback.

两个后端保持同一契约（NCHW float32 进，原始输出张量列表出），这样流水线
在 x86 上开发调试、在 RK3588S 上跑 NPU 用的是同一份上层代码。
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class BaseBackend:
    name = "base"
    input_layout = "nchw"

    def infer(self, tensor: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError

    def adapt(self, hwc: np.ndarray) -> np.ndarray:
        """HWC float32 [0,1] -> 后端需要的输入张量。"""
        if self.input_layout == "nhwc":
            return np.ascontiguousarray(hwc[None])
        return np.ascontiguousarray(hwc.transpose(2, 0, 1)[None])

    def close(self) -> None:
        pass


class OrtBackend(BaseBackend):
    name = "onnxruntime-cpu"
    input_layout = "nchw"

    def __init__(self, model_path: str, threads: int = 4) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input = self.session.get_inputs()[0].name

    def infer(self, tensor: np.ndarray) -> List[np.ndarray]:
        return self.session.run(None, {self._input: tensor})


class RknnBackend(BaseBackend):
    name = "rknn-npu"
    # rknnlite 对 4D 输入默认按 NHWC 解释；直接喂 NHWC 省掉一次内部重排
    input_layout = "nhwc"

    def __init__(self, model_path: str, core_mask: int | None = None) -> None:
        from rknnlite.api import RKNNLite

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        kwargs = {"core_mask": core_mask} if core_mask is not None else {}
        ret = self.rknn.init_runtime(**kwargs)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

    def infer(self, tensor: np.ndarray) -> List[np.ndarray]:
        outs = self.rknn.inference(inputs=[tensor], data_format=["nhwc"])
        if outs is None:
            raise RuntimeError("RKNN inference returned None")
        return [np.asarray(o) for o in outs]

    def close(self) -> None:
        try:
            self.rknn.release()
        except Exception:
            pass


def build_backend(kind: str, model_path: str, **kwargs) -> BaseBackend:
    if kind in ("rknn", "npu"):
        return RknnBackend(model_path, kwargs.get("core_mask"))
    return OrtBackend(model_path, kwargs.get("threads", 4))


def letterbox(
    img: np.ndarray, new_shape: int = 640, color: Sequence[int] = (114, 114, 114)
):
    """等比缩放 + 灰边填充，返回 (图, ratio, (dw, dh))。"""
    import cv2

    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    dh, dw = (new_shape - nh) / 2, (new_shape - nw) / 2
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)
