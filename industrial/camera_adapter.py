"""آداپتر دوربین صنعتی — GigE Vision / RTSP / Mock
برای محیط تولید، harvesters + GenICam استفاده می‌شود؛ برای تست Mock.
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from PIL import Image

from tvdi_industrial.config import get_settings

logger = logging.getLogger(__name__)


class CameraAdapter(ABC):
    @abstractmethod
    def open(self): ...
    @abstractmethod
    def capture(self) -> np.ndarray: ...
    @abstractmethod
    def close(self): ...


class MockCamera(CameraAdapter):
    """دوربین Mock برای CI و توسعه"""

    def __init__(self, defect_type="good", size=128, seed=42):
        self._rng = np.random.default_rng(seed)
        self._size = size
        self._defect_type = defect_type

    def open(self): ...

    def capture(self) -> np.ndarray:
        img = self._rng.normal(128, 8, (self._size, self._size))
        if self._defect_type == "bubble":
            cx, cy = self._rng.integers(20, self._size - 20, 2)
            r = 10
            img[cy - r:cy + r, cx - r:cx + r] = 220
        return np.clip(img, 0, 255).astype(np.uint8)

    def close(self): ...


class RTSPCamera(CameraAdapter):
    """دوربین IP از طریق OpenCV"""

    def __init__(self, url: str):
        self.url = url
        self._cap = None

    def open(self):
        import cv2
        self._cap = cv2.VideoCapture(self.url)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open RTSP: {self.url}")

    def capture(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("frame grab failed")
        return frame

    def close(self):
        if self._cap:
            self._cap.release()


class GigEVisionCamera(CameraAdapter):
    """دوربین GigE Vision با harvesters (نیازمند نصب harvesters + GenTL)"""

    def __init__(self, cti_path: str, serial: Optional[str] = None):
        self.cti_path = cti_path
        self.serial = serial
        self._harvester = None
        self._ia = None

    def open(self):
        try:
            from harvesters.core import Harvester
        except ImportError as exc:
            raise RuntimeError("install harvesters: pip install harvesters") from exc
        h = Harvester()
        h.add_file(self.cti_path)
        h.update()
        ia = h.create({'serial_number': self.serial} if self.serial else {})
        ia.start()
        self._harvester, self._ia = h, ia

    def capture(self) -> np.ndarray:
        with self._ia.fetch() as buffer:
            component = buffer.payload.components[0]
            return component.data.reshape(component.height, component.width)

    def close(self):
        if self._ia:
            self._ia.stop()
            self._ia.destroy()
        if self._harvester:
            self._harvester.reset()


def build_camera(kind: str = "mock", **kwargs) -> CameraAdapter:
    if kind == "mock":
        return MockCamera(**kwargs)
    if kind == "rtsp":
        return RTSPCamera(**kwargs)
    if kind == "gige":
        return GigEVisionCamera(**kwargs)
    raise ValueError(f"unknown camera: {kind}")
