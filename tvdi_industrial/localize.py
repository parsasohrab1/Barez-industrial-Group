"""Box the marked X-ray defect instead of returning only an image class.

Official split of Zenodo 11381120. The score threshold is chosen on the
validation split. The test split is scored once. Coordinates are normalized
to the image: cx, cy, w, h.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from tvdi_industrial.features import extract_features, features_to_vector
from tvdi_industrial.real_xray import CLASS_FA, CLASS_NAMES, ROOT, SOURCE

MODEL_PATH = Path("models/xray_localizer.joblib")
REPORT_PATH = Path("data/real/localization-report.json")
CANVAS = 128
SCALES = ((32, 16), (64, 32))
# Fixed before the official test score is computed.
IOU_HIT = 0.30
GATE_RECALL = 0.40
GATE_PRECISION = 0.25
BACKGROUND = 4


def yolo_boxes(label_path: Path) -> list[tuple]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id, cx, cy, width, height = parts[:5]
        boxes.append((int(class_id), float(cx), float(cy), float(width), float(height)))
    return boxes


def box_iou(a, b) -> float:
    ax0, ay0 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax1, ay1 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx1, by1 = b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    if union <= 0:
        return 0.0
    return inter / union


def _crop(image: np.ndarray, cx: float, cy: float, width: float, height: float) -> np.ndarray:
    side_y, side_x = image.shape
    x0 = int(np.clip((cx - width / 2) * side_x, 0, side_x - 1))
    x1 = int(np.clip((cx + width / 2) * side_x, x0 + 1, side_x))
    y0 = int(np.clip((cy - height / 2) * side_y, 0, side_y - 1))
    y1 = int(np.clip((cy + height / 2) * side_y, y0 + 1, side_y))
    crop = image[y0:y1, x0:x1]
    return np.asarray(Image.fromarray(crop).resize((48, 48)), dtype=np.float32)


def _windows(image: np.ndarray):
    found = []
    for window, stride in SCALES:
        limit = CANVAS - window + 1
        for y in range(0, limit, stride):
            for x in range(0, limit, stride):
                crop = image[y:y + window, x:x + window]
                found.append((
                    (x + window / 2) / CANVAS,
                    (y + window / 2) / CANVAS,
                    window / CANVAS,
                    window / CANVAS,
                    np.asarray(Image.fromarray(crop).resize((48, 48)), dtype=np.float32),
                ))
    return found


def _load_canvas(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((CANVAS, CANVAS))
    return np.asarray(image)


def _match(predictions, truths) -> tuple[int, int, int]:
    hits = 0
    used = set()
    ious = []
    for truth in truths:
        best_iou = 0.0
        best_index = None
        for index, pred in enumerate(predictions):
            if index in used or pred["class_id"] != truth[0]:
                continue
            overlap = box_iou((pred["cx"], pred["cy"], pred["w"], pred["h"]), truth[1:])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None and best_iou >= IOU_HIT:
            hits += 1
            used.add(best_index)
            ious.append(best_iou)
    return hits, len(predictions), len(truths)


def _nms(boxes: list[dict], iou_limit: float = 0.40) -> list[dict]:
    ordered = sorted(boxes, key=lambda item: item["score"], reverse=True)
    kept = []
    for box in ordered:
        if all(box_iou(
            (box["cx"], box["cy"], box["w"], box["h"]),
            (other["cx"], other["cy"], other["w"], other["h"]),
        ) < iou_limit for other in kept):
            kept.append(box)
        if len(kept) == 6:
            break
    return kept


def _propose(canvas: np.ndarray, model) -> list[dict]:
    windows = _windows(canvas)
    if not windows:
        return []
    vectors = np.vstack([
        features_to_vector(extract_features(crop)) for *_box, crop in windows
    ])
    probabilities = model.predict_proba(vectors)
    proposals = []
    for (cx, cy, width, height, _crop), proba in zip(windows, probabilities):
        defect_scores = proba[:BACKGROUND]
        class_id = int(np.argmax(defect_scores))
        proposals.append({
            "class_id": class_id,
            "class": CLASS_NAMES[class_id],
            "label_fa": CLASS_FA[CLASS_NAMES[class_id]],
            "score": float(defect_scores[class_id]),
            "cx": float(cx),
            "cy": float(cy),
            "w": float(width),
            "h": float(height),
        })
    return proposals


def locate(canvas: np.ndarray, bundle: dict) -> list[dict]:
    proposals = _propose(canvas, bundle["model"])
    chosen = [box for box in proposals if box["score"] >= bundle["threshold"]]
    return _nms(chosen)


def _score_split(split: str, model, threshold: float, root: Path) -> dict:
    hits = predictions = truths = 0
    matched_ious = []
    image_dir = root / split / "images"
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = root / split / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        boxes = locate(_load_canvas(image_path), {"model": model, "threshold": threshold})
        truth = yolo_boxes(label_path)
        hit, pred_count, truth_count = _match(boxes, truth)
        hits += hit
        predictions += pred_count
        truths += truth_count
    recall = hits / truths if truths else 0.0
    precision = hits / predictions if predictions else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "hits": hits,
        "predictions": predictions,
        "ground_truth_boxes": truths,
        "recall_iou_0_30": recall,
        "precision_iou_0_30": precision,
        "f1_iou_0_30": f1,
    }


def _training_matrix(root: Path, rng: np.random.Generator):
    vectors = []
    labels = []
    image_dir = root / "train" / "images"
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = root / "train" / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        canvas = _load_canvas(image_path)
        truths = yolo_boxes(label_path)
        for class_id, cx, cy, width, height in truths:
            crop = _crop(canvas, cx, cy, max(width, 0.08), max(height, 0.08))
            vectors.append(features_to_vector(extract_features(crop)))
            labels.append(class_id)
        for _ in range(max(1, len(truths))):
            for _try in range(8):
                width = float(rng.uniform(0.12, 0.28))
                height = float(rng.uniform(0.12, 0.28))
                cx = float(rng.uniform(width / 2, 1 - width / 2))
                cy = float(rng.uniform(height / 2, 1 - height / 2))
                candidate = (cx, cy, width, height)
                if all(box_iou(candidate, truth[1:]) < 0.05 for truth in truths):
                    crop = _crop(canvas, cx, cy, width, height)
                    vectors.append(features_to_vector(extract_features(crop)))
                    labels.append(BACKGROUND)
                    break
        if len(labels) % 800 < 4 and len(labels) > 400:
            print(f"  crops: {len(labels)}", flush=True)
    return np.vstack(vectors), np.array(labels)


def _choose_threshold(model, root: Path) -> tuple[float, dict]:
    cached = []
    image_dir = root / "valid" / "images"
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = root / "valid" / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        cached.append((yolo_boxes(label_path), _propose(_load_canvas(image_path), model)))
        if len(cached) % 100 == 0:
            print(f"  valid windows: {len(cached)}", flush=True)
    best_threshold = 0.55
    best_f1 = -1.0
    best = {}
    for threshold in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        hits = predictions = truths = 0
        for truth, proposals in cached:
            chosen = _nms([box for box in proposals if box["score"] >= threshold])
            hit, pred_count, truth_count = _match(chosen, truth)
            hits += hit
            predictions += pred_count
            truths += truth_count
        recall = hits / truths if truths else 0.0
        precision = hits / predictions if predictions else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        print(f"  valid threshold {threshold:.2f} recall {recall:.3f} precision {precision:.3f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best = {
                "threshold": threshold,
                "recall_iou_0_30": recall,
                "precision_iou_0_30": precision,
                "f1_iou_0_30": f1,
            }
    return best_threshold, best


def run_localization(root: Path = ROOT, publish: bool = True) -> dict:
    rng = np.random.default_rng(42)
    print("training localizer on train crops", flush=True)
    started = time.perf_counter()
    vectors, labels = _training_matrix(root, rng)
    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(vectors, labels)
    print(f"crops {len(labels)} fit {time.perf_counter() - started:.1f}s", flush=True)
    threshold, validation = _choose_threshold(model, root)
    print("scoring official test once", flush=True)
    test = _score_split("test", model, threshold, root)
    gate_passed = (
        test["recall_iou_0_30"] >= GATE_RECALL
        and test["precision_iou_0_30"] >= GATE_PRECISION
    )
    report = {
        "task": "normalized box of the marked structural defect",
        "gate_passed": gate_passed,
        "dataset": SOURCE,
        "iou_hit": IOU_HIT,
        "gates": {"recall_at_least": GATE_RECALL, "precision_at_least": GATE_PRECISION},
        "chosen_on_validation": validation,
        "test": test,
        "coordinates": "cx, cy, w, h normalized to the image",
        "not_a_plant_claim": "Boxes are on the public Shanghai X-ray set, not a Barez tire.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if publish:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "threshold": threshold, "classes": CLASS_NAMES}, MODEL_PATH)
    return report


def predict_regions(image_bytes: bytes) -> list[dict]:
    if not MODEL_PATH.exists():
        return []
    bundle = joblib.load(MODEL_PATH)
    image = Image.open(__import__("io").BytesIO(image_bytes)).convert("L").resize((CANVAS, CANVAS))
    boxes = locate(np.asarray(image), bundle)
    for box in boxes:
        box.pop("class_id", None)
    return boxes
