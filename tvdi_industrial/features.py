"""Hand-crafted image features used by the fallback classifier."""
import numpy as np

IMAGE_SIZE = 128
CLASSES = ["good", "bubble", "tread_uneven", "layer_separation"]
FEATURE_NAMES = [
    "mean", "std", "min", "max", "p10", "p50", "p90",
    "dark_ratio", "bright_ratio", "grad_mean", "grad_std",
    "lap_var", "edge_density",
]


def extract_features(image) -> dict:
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)

    gy, gx = np.gradient(img)
    grad_mag = np.hypot(gx, gy)
    lap = (
        -4 * img
        + np.roll(img, 1, 0)
        + np.roll(img, -1, 0)
        + np.roll(img, 1, 1)
        + np.roll(img, -1, 1)
    )
    return {
        "mean": float(img.mean()),
        "std": float(img.std()),
        "min": float(img.min()),
        "max": float(img.max()),
        "p10": float(np.percentile(img, 10)),
        "p50": float(np.percentile(img, 50)),
        "p90": float(np.percentile(img, 90)),
        "dark_ratio": float((img < 50).mean()),
        "bright_ratio": float((img > 200).mean()),
        "grad_mean": float(grad_mag.mean()),
        "grad_std": float(grad_mag.std()),
        "lap_var": float(lap.var()),
        "edge_density": float((grad_mag > 20).mean()),
    }


def features_to_vector(features: dict) -> np.ndarray:
    return np.array([features[name] for name in FEATURE_NAMES], dtype=float)
