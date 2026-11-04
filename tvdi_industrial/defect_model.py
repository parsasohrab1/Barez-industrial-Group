"""Joblib classifier with the same heuristic fallback as the ONNX path."""
from pathlib import Path

import joblib
import numpy as np

from tvdi_industrial.config import get_settings
from tvdi_industrial.features import features_to_vector


class DefectModel:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def load(cls, path=None):
        settings = get_settings()
        path = Path(path or settings.model_fallback_path)
        if path.exists():
            return cls(joblib.load(path))
        return cls(None)

    def predict(self, features: dict) -> dict:
        if self.model is not None:
            vector = features_to_vector(features).reshape(1, -1)
            proba = self.model.predict_proba(vector)[0]
            classes = list(self.model.classes_)
            idx = int(np.argmax(proba))
            p_good = proba[classes.index("good")] if "good" in classes else 0.0
            return {
                "defect_type": classes[idx],
                "defect_probability": float(1.0 - p_good),
                "confidence": float(proba[idx]),
                "source": "joblib",
            }

        bright = features["bright_ratio"]
        dark = features["dark_ratio"]
        grad_std = features["grad_std"]
        edge_density = features["edge_density"]
        lap_var = features["lap_var"]

        if bright > 0.008 or lap_var > 250:
            defect_type = "bubble"
            score = min(0.95, 0.55 + bright * 10 + lap_var / 2000)
        elif grad_std > 22 or edge_density > 0.18:
            defect_type = "tread_uneven"
            score = min(0.95, 0.55 + grad_std / 100 + edge_density / 2)
        elif dark > 0.012:
            defect_type = "layer_separation"
            score = min(0.95, 0.60 + dark * 12)
        else:
            defect_type = "good"
            score = max(0.01, min(0.40, bright * 20 + dark * 20 + grad_std / 200))

        return {
            "defect_type": defect_type,
            "defect_probability": float(score),
            "confidence": float(score),
            "source": "heuristic",
        }
