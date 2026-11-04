"""Full-image software cycle versus a single patch inference.

Sixteen vertical strips stand in for one revolution of sectors. The clock
measures only computation on this machine. Mechanical rotation and the X-ray
gantry are not measured here.
"""
import time

import numpy as np
from PIL import Image

SECTORS = 16
# Poling TVIS publishes a three-laser scan at 60 rpm: one revolution is one second.
PUBLISHED_APPEARANCE_REVOLUTION_S = 1.0
# Micro-Poise VerTiX-PLUS publishes a full-tire X-ray cycle under 22 seconds.
PUBLISHED_XRAY_MACHINE_CYCLE_S = 22.0


def software_cycle(image, score_sector) -> dict:
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    height, width = gray.shape[:2]
    started = time.perf_counter()
    for index in range(SECTORS):
        x0 = int(index * width / SECTORS)
        x1 = max(x0 + 1, int((index + 1) * width / SECTORS))
        strip = gray[:, x0:x1]
        canvas = Image.fromarray(np.asarray(strip, dtype=np.uint8)).resize((128, 128))
        score_sector(canvas)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "sectors": SECTORS,
        "software_cycle_ms": float(elapsed_ms),
        "per_sector_ms": float(elapsed_ms / SECTORS),
        "motion_measured": False,
        "published_reference": {
            "appearance_revolution_s": PUBLISHED_APPEARANCE_REVOLUTION_S,
            "appearance_source": "Poling TVIS operating point, 60 rpm",
            "xray_machine_cycle_claim_s": PUBLISHED_XRAY_MACHINE_CYCLE_S,
            "xray_source": "Micro-Poise VerTiX-PLUS published cycle",
        },
        "note": "software_cycle_ms is compute on this PC. It is not a tire-machine cycle.",
    }
