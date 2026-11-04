"""Relevant-environment check on a public tire X-ray set.

Dataset: Shanghai University Network Control Centre, Zenodo 11381120,
CC BY 4.0. Official train/valid/test split. This is not Barez plant data.
The test split is scored once, after the feature set is chosen on validation.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from tvdi_industrial.features import extract_features, features_to_vector

ROOT = Path("data/real/xray")
MODEL_PATH = Path("models/xray_defect.joblib")
REPORT_PATH = Path("data/real/trl5-report.json")
CLASS_NAMES = ["open", "cords_defects", "impurity", "belt_cuobian"]
CLASS_FA = {
    "open": "بازشدگی اتصال",
    "cords_defects": "عیب کورد",
    "impurity": "ناخالصی",
    "belt_cuobian": "جابه‌جایی کمربند",
}
# Gates fixed before looking at the official test scores.
GATE_MACRO_F1 = 0.60
GATE_ACCURACY_MARGIN = 0.15
SOURCE = {
    "title": "Tyre defect data set",
    "doi": "10.5281/zenodo.11381120",
    "license": "CC BY 4.0",
    "producer": "Shanghai University Network Control Centre",
    "url": "https://zenodo.org/records/11381120",
}


def image_label(label_path: Path) -> int:
    counts: dict[int, int] = {}
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id = int(line.split()[0])
        counts[class_id] = counts.get(class_id, 0) + 1
    if not counts:
        raise ValueError(f"empty label: {label_path}")
    return max(counts, key=counts.get)


def iter_split(split: str, root: Path = ROOT):
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        yield image_path, image_label(label_path)


def _resize(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size))
    return np.asarray(image, dtype=np.float32)


def global_features(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((128, 128))
    return features_to_vector(extract_features(image))


def hog_features(path: Path) -> np.ndarray:
    image = _resize(path, 96) / 255.0
    gy, gx = np.gradient(image)
    magnitude = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + np.pi) * (9 / (2 * np.pi))
    angle = np.clip(angle, 0, 8.999)
    cell = 8
    cells = []
    for y in range(0, 96, cell):
        row = []
        for x in range(0, 96, cell):
            histogram = np.zeros(9, dtype=np.float32)
            bins = angle[y:y + cell, x:x + cell].astype(np.int32).ravel()
            weights = magnitude[y:y + cell, x:x + cell].ravel()
            np.add.at(histogram, bins, weights)
            row.append(histogram)
        cells.append(row)
    blocks = []
    for y in range(len(cells) - 1):
        for x in range(len(cells[0]) - 1):
            block = np.concatenate([
                cells[y][x], cells[y][x + 1], cells[y + 1][x], cells[y + 1][x + 1],
            ])
            blocks.append(block / (np.linalg.norm(block) + 1e-6))
    return np.concatenate(blocks).astype(np.float32)


def _matrix(split: str, feature_fn, root: Path):
    vectors, labels, paths = [], [], []
    started = time.perf_counter()
    for path, label in iter_split(split, root):
        vectors.append(feature_fn(path))
        labels.append(label)
        paths.append(path.name)
        if len(labels) % 400 == 0:
            print(f"  {split}: {len(labels)}", flush=True)
    elapsed = time.perf_counter() - started
    return np.vstack(vectors), np.array(labels), paths, elapsed


def _scores(y_true, y_pred) -> dict:
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0,
    )
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        row = report[str(index)]
        per_class[name] = {
            "label_fa": CLASS_FA[name],
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1-score"]),
            "support": int(row["support"]),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": per_class,
    }


def run_trl5(root=ROOT, publish: bool = True) -> dict:
    root = Path(root)
    if not (root / "train" / "images").exists():
        raise FileNotFoundError(
            "Real X-ray set is missing. Expected data/real/xray from Zenodo 11381120."
        )
    print("extracting features", flush=True)
    candidates = {
        "global-13": global_features,
        "hog": hog_features,
    }
    valid_f1 = {}
    fitted = {}
    for name, feature_fn in candidates.items():
        print(name, flush=True)
        x_train, y_train, _, _ = _matrix("train", feature_fn, root)
        x_valid, y_valid, _, _ = _matrix("valid", feature_fn, root)
        if name == "global-13":
            model = RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced",
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.08, max_depth=8, random_state=42,
            )
        model.fit(x_train, y_train)
        scores = _scores(y_valid, model.predict(x_valid))
        valid_f1[name] = scores["macro_f1"]
        fitted[name] = (model, feature_fn, x_train, y_train)
        print(f"  valid macro F1 {scores['macro_f1']:.3f}", flush=True)

    chosen = max(valid_f1, key=valid_f1.get)
    model, feature_fn, x_train, y_train = fitted[chosen]
    print("official test with", chosen, flush=True)
    x_test, y_test, names, feature_seconds = _matrix("test", feature_fn, root)
    started = time.perf_counter()
    prediction = model.predict(x_test)
    predict_ms = (time.perf_counter() - started) * 1000 / max(len(y_test), 1)
    test_scores = _scores(y_test, prediction)

    majority = int(np.bincount(y_train).argmax())
    baseline_accuracy = float(np.mean(y_test == majority))
    passed = (
        test_scores["macro_f1"] >= GATE_MACRO_F1
        and test_scores["accuracy"] >= baseline_accuracy + GATE_ACCURACY_MARGIN
    )
    report = {
        "trl": 5 if passed else 4,
        "gate_passed": passed,
        "environment": "relevant public tire X-ray images, not the Barez line",
        "dataset": SOURCE,
        "task": "image-level type of a marked structural defect",
        "classes": CLASS_NAMES,
        "labels_fa": CLASS_FA,
        "chosen_on_validation": chosen,
        "validation_macro_f1": valid_f1,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "majority_class": CLASS_NAMES[majority],
        "majority_baseline_accuracy": baseline_accuracy,
        "gates": {
            "macro_f1_at_least": GATE_MACRO_F1,
            "accuracy_margin_over_majority": GATE_ACCURACY_MARGIN,
        },
        "metrics": test_scores,
        "latency_ms": {
            "predict_mean": float(predict_ms),
            "feature_seconds_test": float(feature_seconds),
        },
        "not_validated": [
            "Barez production tires",
            "process traceability",
            "OPC UA and MQTT",
            "uniformity, balance, and shearography",
            "appearance defects on a rotating tire",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "examples": [
            {"file": names[i], "true": CLASS_NAMES[int(y_test[i])], "predicted": CLASS_NAMES[int(prediction[i])]}
            for i in range(min(8, len(names)))
        ],
    }
    if publish:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": chosen, "classes": CLASS_NAMES}, MODEL_PATH)
    return report


def predict_xray(image_bytes: bytes) -> dict:
    bundle = joblib.load(MODEL_PATH)
    feature_fn = hog_features if bundle["features"] == "hog" else global_features
    temporary = Path("data/real/_upload.jpg")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(image_bytes)
    vector = feature_fn(temporary).reshape(1, -1)
    model = bundle["model"]
    class_index = int(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    name = CLASS_NAMES[class_index]
    from tvdi_industrial.localize import MODEL_PATH as LOCALIZER_PATH
    from tvdi_industrial.localize import predict_regions

    regions = predict_regions(image_bytes) if LOCALIZER_PATH.exists() else []
    return {
        "defect_type": name,
        "label_fa": CLASS_FA[name],
        "confidence": float(probabilities[class_index]),
        "source": "xray-zenodo-11381120",
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
        },
        "regions": regions,
        "coordinate_space": "normalized cx cy w h",
    }
