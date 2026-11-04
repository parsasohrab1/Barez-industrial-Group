"""Synthetic images whose process metadata actually correlates with the label.

The pictures are still drawings, not Barez plant images. The correlation exists
so traceability code can be tested without inventing a random lift.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from tvdi_industrial.features import CLASSES, IMAGE_SIZE

MACHINE_BIAS = {
    "M01": "bubble",
    "M02": "tread_uneven",
    "M03": "layer_separation",
    "M04": "good",
    "M05": "bubble",
}
LOT_BIAS = {
    "bubble": "L01",
    "tread_uneven": "L02",
    "layer_separation": "L03",
    "good": "L04",
}


def _base_image(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(128, 8, (IMAGE_SIZE, IMAGE_SIZE))


def _add_bubble(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    cx, cy = rng.integers(20, IMAGE_SIZE - 20, 2)
    radius = int(rng.integers(6, 18))
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=220)
    return np.asarray(pil, dtype=float)


def _add_tread_uneven(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    _, x = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    period = int(rng.integers(8, 20))
    amplitude = float(rng.uniform(10, 25))
    phase = float(rng.uniform(0, 2 * np.pi))
    return img + amplitude * np.sin(2 * np.pi * x / period + phase)


def _add_layer_separation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    x1, y1 = rng.integers(0, IMAGE_SIZE, 2)
    x2, y2 = rng.integers(0, IMAGE_SIZE, 2)
    draw.line([int(x1), int(y1), int(x2), int(y2)], fill=25, width=int(rng.integers(2, 6)))
    return np.asarray(pil, dtype=float)


def make_image(defect_type: str, rng: np.random.Generator) -> np.ndarray:
    img = _base_image(rng)
    if defect_type == "bubble":
        img = _add_bubble(img, rng)
    elif defect_type == "tread_uneven":
        img = _add_tread_uneven(img, rng)
    elif defect_type == "layer_separation":
        img = _add_layer_separation(img, rng)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _machine_for(defect_type: str, rng: np.random.Generator) -> str:
    preferred = [machine for machine, bias in MACHINE_BIAS.items() if bias == defect_type]
    if preferred and rng.random() < 0.75:
        return str(rng.choice(preferred))
    return f"M{int(rng.integers(1, 6)):02d}"


def _process_for(defect_type: str, rng: np.random.Generator) -> dict:
    values = {
        "build_temp": float(rng.uniform(150, 165)),
        "build_pressure": float(rng.uniform(10, 13)),
        "cure_temp": float(rng.uniform(158, 172)),
        "cure_time": float(rng.uniform(13, 18)),
    }
    if defect_type == "bubble":
        values["build_temp"] = float(rng.uniform(168.5, 172))
    elif defect_type == "tread_uneven":
        values["build_pressure"] = float(rng.uniform(8.0, 9.3))
    elif defect_type == "layer_separation":
        values["cure_time"] = float(rng.uniform(10.0, 11.7))
    return values


def specimen_metadata(defect_type: str, rng: np.random.Generator, tire_id: str) -> dict:
    """Process metadata for one laboratory specimen, correlated with its label."""
    return {
        "tire_id": tire_id,
        "batch_id": f"B{int(rng.integers(1, 9)):03d}",
        "machine_id": _machine_for(defect_type, rng),
        "mold_id": f"D{int(rng.integers(1, 9)):02d}",
        "material_lot": LOT_BIAS[defect_type],
        **_process_for(defect_type, rng),
        "defect_type": defect_type,
    }


def generate_dataset(output_dir, n_per_class: int = 50, seed: int = 42) -> pd.DataFrame:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []

    for defect_type in CLASSES:
        for i in range(n_per_class):
            tire_id = f"TIRE-{defect_type[:3].upper()}-{i:04d}"
            Image.fromarray(make_image(defect_type, rng)).save(images_dir / f"{tire_id}.png")
            machine_id = _machine_for(defect_type, rng)
            rows.append({
                "tire_id": tire_id,
                "batch_id": f"B{int(rng.integers(1, 9)):03d}",
                "machine_id": machine_id,
                "mold_id": f"D{int(rng.integers(1, 9)):02d}",
                "material_lot": LOT_BIAS[defect_type],
                **_process_for(defect_type, rng),
                "defect_type": defect_type,
                "defect_probability": float(0.05 if defect_type == "good" else rng.uniform(0.55, 0.98)),
                "severity": (
                    "high" if defect_type == "layer_separation"
                    else "medium" if defect_type in ("bubble", "tread_uneven")
                    else "low"
                ),
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "metadata.csv", index=False)
    return frame
