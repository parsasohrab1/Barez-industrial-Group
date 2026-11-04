"""Drift Detection ساده بر اساس KS-test و فاصله توزیع"""
import json
from pathlib import Path
from typing import Optional

import numpy as np


class DriftDetector:
    def __init__(self, baseline_path: Optional[str] = None, threshold: float = 0.15):
        self.baseline: Optional[np.ndarray] = None
        self.threshold = threshold
        if baseline_path and Path(baseline_path).exists():
            self.baseline = np.array(json.loads(Path(baseline_path).read_text())["features"])

    def fit_baseline(self, features: np.ndarray):
        self.baseline = features

    def score(self, current: np.ndarray) -> float:
        if self.baseline is None:
            return 0.0
        try:
            from scipy.stats import ks_2samp
            scores = []
            for col in range(current.shape[1]):
                _, p = ks_2samp(self.baseline[:, col], current[:, col])
                scores.append(1 - p)
            return float(np.mean(scores))
        except ImportError:
            # fallback: relative mean shift
            base_mean = self.baseline.mean(axis=0)
            cur_mean = current.mean(axis=0)
            return float(np.abs((cur_mean - base_mean) / (np.abs(base_mean) + 1e-9)).mean())

    def is_drifted(self, current: np.ndarray) -> bool:
        return self.score(current) > self.threshold
