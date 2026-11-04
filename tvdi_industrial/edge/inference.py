"""Edge Inference با ONNX Runtime + fallback heuristic"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from tvdi_industrial.config import get_settings

logger = logging.getLogger(__name__)

CLASSES = ["good", "bubble", "tread_uneven", "layer_separation"]
IMAGE_SIZE = 128


class ONNXInference:
    def __init__(self, model_path: Optional[Path] = None, providers=None):
        self.s = get_settings()
        self.model_path = Path(model_path or self.s.model_path)
        self._session = None
        self._providers = providers or ["CPUExecutionProvider"]

    def load(self):
        if not self.model_path.exists():
            logger.warning("ONNX model missing: %s — using heuristic", self.model_path)
            return
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed — using heuristic")
            return
        self._session = ort.InferenceSession(
            str(self.model_path), providers=self._providers
        )
        logger.info("ONNX loaded: %s providers=%s", self.model_path, self._session.get_providers())

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = image.mean(axis=2)
        img = Image.fromarray(image.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr[None, None, :, :]  # N,C,H,W

    def predict(self, image: np.ndarray) -> dict:
        if self._session is not None:
            x = self.preprocess(image)
            out = self._session.run(None, {self._session.get_inputs()[0].name: x})[0]
            probs = _softmax(out[0])
            idx = int(np.argmax(probs))
            return {
                "defect_type": CLASSES[idx],
                "defect_probability": float(1.0 - probs[0]) if CLASSES[0] == "good" else float(probs[idx]),
                "confidence": float(probs[idx]),
                "source": "onnx",
            }
        return self._heuristic(image)

    def _heuristic(self, image: np.ndarray) -> dict:
        img = image.astype(np.float32)
        if img.ndim == 3:
            img = img.mean(axis=2)
        bright = float((img > 200).mean())
        dark = float((img < 50).mean())
        gy, gx = np.gradient(img)
        grad_std = float(np.hypot(gx, gy).std())
        lap = float(((-4 * img + np.roll(img, 1, 0) + np.roll(img, -1, 0)
                     + np.roll(img, 1, 1) + np.roll(img, -1, 1))).var())

        if bright > 0.008 or lap > 250:
            return {"defect_type": "bubble", "defect_probability": min(0.95, 0.55 + bright * 10),
                    "confidence": 0.7, "source": "heuristic"}
        if grad_std > 22:
            return {"defect_type": "tread_uneven", "defect_probability": min(0.95, 0.55 + grad_std / 100),
                    "confidence": 0.7, "source": "heuristic"}
        if dark > 0.012:
            return {"defect_type": "layer_separation", "defect_probability": min(0.95, 0.6 + dark * 12),
                    "confidence": 0.7, "source": "heuristic"}
        return {"defect_type": "good", "defect_probability": 0.05,
                "confidence": 0.9, "source": "heuristic"}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
