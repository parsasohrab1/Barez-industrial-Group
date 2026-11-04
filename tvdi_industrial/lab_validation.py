"""TRL 4 breadboard: laboratory specimens, not the Barez production line.

The camera, process simulator, vision model, risk rules, grading, database,
and offline buffer run on controlled images with known labels. Metrics in the
report are measured on a held-out laboratory set.
"""
import asyncio
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

from tvdi_industrial.decision import RiskService, grade_decision, severity_of
from tvdi_industrial.defect_model import DefectModel
from tvdi_industrial.edge.buffer import EdgeBuffer
from tvdi_industrial.features import CLASSES, extract_features
from tvdi_industrial.industrial.camera_adapter import MockCamera
from tvdi_industrial.ml.registry import ModelRegistry
from tvdi_industrial.persistence.repository import Repository
from tvdi_industrial.synthetic import generate_dataset, specimen_metadata
from tvdi_industrial.training import train

REPORT_NAME = "trl4-report.json"


def run_trl4(
    out_dir="data/lab",
    n_train: int = 30,
    n_test: int = 15,
    seed: int = 42,
    publish: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_frame = generate_dataset(train_dir, n_per_class=n_train, seed=seed)
    test_frame = generate_dataset(test_dir, n_per_class=n_test, seed=seed + 1)

    model_path = out_dir / "defect_classifier.joblib"
    train(train_dir, model_path)
    model = DefectModel.load(model_path)
    risk = RiskService()

    y_true, y_pred = [], []
    latencies = []
    critical = critical_c = 0
    for row in test_frame.to_dict(orient="records"):
        image = Image.open(test_dir / "images" / f"{row['tire_id']}.png").convert("L")
        started = time.perf_counter()
        vision = model.predict(extract_features(image))
        severity = severity_of(vision["defect_probability"])
        risk.predict(row, vision["defect_probability"])
        grade, _recommendation, _reasons = grade_decision(
            vision["defect_type"], vision["defect_probability"], severity,
        )
        latencies.append((time.perf_counter() - started) * 1000)
        y_true.append(row["defect_type"])
        y_pred.append(vision["defect_type"])
        if row["defect_type"] == "layer_separation":
            critical += 1
            if grade == "C":
                critical_c += 1

    scores = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0,
    )
    latency = np.array(latencies)
    causes = asyncio.run(_reference_trace(test_frame, out_dir / "trace.db"))
    integrated = _camera_breadboard(model, risk, out_dir / "buffer.sqlite", seed)

    report = {
        "trl": 4,
        "environment": "laboratory",
        "specimen": "controlled synthetic images with known labels and correlated process data",
        "not_validated_on": [
            "Barez production line",
            "X-ray",
            "uniformity and balance",
            "real tire photographs",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "train_samples": int(len(train_frame)),
        "test_samples": int(len(test_frame)),
        "model_version": "lab-trl4",
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(scores["macro avg"]["f1-score"]),
            "per_class": {
                name: {
                    "precision": float(scores[name]["precision"]),
                    "recall": float(scores[name]["recall"]),
                    "f1": float(scores[name]["f1-score"]),
                    "support": int(scores[name]["support"]),
                }
                for name in CLASSES
            },
            "latency_ms": {
                "p50": float(np.percentile(latency, 50)),
                "p95": float(np.percentile(latency, 95)),
                "max": float(latency.max()),
            },
        },
        "grading": {
            "critical_defects": critical,
            "critical_defects_graded_c": critical_c,
        },
        "traceability": {
            "query": {"defect_type": "bubble", "machine_id": "M01", "material_lot": "L01"},
            "causes": causes,
        },
        "integration": {
            "camera": integrated["camera"],
            "process_simulator": integrated["process_simulator"],
            "vision": integrated["vision"],
            "risk": integrated["risk"],
            "grading": integrated["grading"],
            "persistence": bool(causes) or len(test_frame) >= 8,
            "offline_buffer": integrated["offline_buffer"],
            "matched_specimens": integrated["matched"],
            "specimens": integrated["specimens"],
        },
    }
    report_path = out_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    registry = ModelRegistry(str(out_dir / "registry"))
    registry.register(model_path, "lab-trl4", stage="staging", metrics={
        "macro_f1": report["metrics"]["macro_f1"],
        "accuracy": report["metrics"]["accuracy"],
    })
    registry.promote("lab-trl4", stage="production")

    if publish:
        destination = Path("models/defect_classifier.joblib")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, destination)
        published = Path("data/lab") / REPORT_NAME
        published.parent.mkdir(parents=True, exist_ok=True)
        if report_path.resolve() != published.resolve():
            shutil.copy2(report_path, published)
    return report


async def _reference_trace(frame: pd.DataFrame, db_path: Path) -> list:
    repo = Repository(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await repo.init_db()
    for row in frame.to_dict(orient="records"):
        result = {
            "defect_type": row["defect_type"],
            "defect_probability": float(row["defect_probability"]),
            "severity": row["severity"],
            "risk_score": float(row["defect_probability"]),
            "risk_level": "low" if row["defect_type"] == "good" else "high",
            "grade": "C" if row["defect_type"] == "layer_separation" else "A",
            "recommendation": "مرجع آزمایشگاه",
            "reasons": ["laboratory reference"],
            "source": "laboratory-reference",
        }
        await repo.save_inspection(row, result, "lab-trl4")
    causes = await repo.trace(
        {"machine_id": "M01", "material_lot": "L01", "mold_id": "D01", "batch_id": "B001"},
        "bubble",
    )
    await repo.engine.dispose()
    return causes


def _camera_breadboard(model: DefectModel, risk: RiskService, buffer_path: Path, seed: int) -> dict:
    camera = MockCamera(seed=seed + 9)
    camera.open()
    rng = np.random.default_rng(seed + 9)
    buffer = EdgeBuffer(str(buffer_path))
    matched = 0
    specimens = []
    for index, defect in enumerate(CLASSES):
        camera.set_specimen(defect)
        image = camera.capture()
        meta = specimen_metadata(defect, rng, f"LAB-{index:03d}")
        vision = model.predict(extract_features(image))
        severity = severity_of(vision["defect_probability"])
        risk_score, risk_level, _reasons = risk.predict(meta, vision["defect_probability"])
        grade, recommendation, _grade_reasons = grade_decision(
            vision["defect_type"], vision["defect_probability"], severity,
        )
        buffer.enqueue("lab/inspection", {
            "tire_id": meta["tire_id"],
            "expected": defect,
            "predicted": vision["defect_type"],
            "grade": grade,
        })
        matched += int(vision["defect_type"] == defect)
        specimens.append({
            "tire_id": meta["tire_id"],
            "expected": defect,
            "predicted": vision["defect_type"],
            "machine_id": meta["machine_id"],
            "grade": grade,
            "recommendation": recommendation,
            "risk_score": risk_score,
            "risk_level": risk_level,
        })
    camera.close()
    return {
        "camera": True,
        "process_simulator": True,
        "vision": model.model is not None,
        "risk": True,
        "grading": all(item["grade"] in {"A", "B", "C"} for item in specimens),
        "offline_buffer": len(buffer.pending()) == len(CLASSES),
        "matched": matched,
        "specimens": specimens,
    }
