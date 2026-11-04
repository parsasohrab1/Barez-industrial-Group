"""
TVDI — Tire Vision Defect Inspector
سامانه هوشمند بازرسی بینایی ماشین و پیش‌بینی عیوب ساخت تایر
برای گروه صنعتی بارز

اجرا:
    python src.py generate --n 50 --out data
    python src.py train --data data --model models/defect_classifier.joblib
    python src.py serve
    python src.py test
"""

# =============================================================================
# 0) IMPORTS
# =============================================================================

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field


# =============================================================================
# 1) CONFIG
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
IMAGES_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.csv"
MODEL_PATH = MODEL_DIR / "defect_classifier.joblib"

IMAGE_SIZE = 128
CLASSES = ["good", "bubble", "tread_uneven", "layer_separation"]


# =============================================================================
# 2) SCHEMAS (Pydantic)
# =============================================================================

class TireMetadata(BaseModel):
    tire_id: str = Field(default="TIRE-0001")
    batch_id: str = "B0001"
    machine_id: str = "M01"
    mold_id: str = "D01"
    material_lot: str = "L01"
    build_temp: float = 155.0
    build_pressure: float = 11.0
    cure_temp: float = 165.0
    cure_time: float = 15.0


class PredictRequest(BaseModel):
    image_base64: str
    metadata: TireMetadata


class Cause(BaseModel):
    field: str
    value: str
    defect_rate: float
    lift: float


class PredictResponse(BaseModel):
    tire_id: str
    defect_type: str
    defect_probability: float
    severity: str
    suspected_causes: List[Cause]
    risk_score: float
    risk_level: str
    grade: str
    recommendation: str
    reasons: List[str]
    features: Optional[Dict[str, float]] = None


# =============================================================================
# 3) FEATURES
# =============================================================================

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


# =============================================================================
# 4) MODELS
# =============================================================================

class DefectModel:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def load(cls, path=MODEL_PATH):
        path = Path(path)
        if path.exists():
            return cls(joblib.load(path))
        return cls(None)

    def predict(self, features: dict) -> dict:
        if self.model is not None:
            x = features_to_vector(features).reshape(1, -1)
            proba = self.model.predict_proba(x)[0]
            classes = list(self.model.classes_)
            idx = int(np.argmax(proba))
            defect_type = classes[idx]
            p_good = proba[classes.index("good")] if "good" in classes else 0.0
            return {
                "defect_type": defect_type,
                "defect_probability": float(1.0 - p_good),
                "confidence": float(proba[idx]),
                "source": "model",
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


# =============================================================================
# 5) SYNTHETIC DATA
# =============================================================================

def _base_image(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(128, 8, (IMAGE_SIZE, IMAGE_SIZE))


def _add_bubble(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    cx, cy = rng.integers(20, IMAGE_SIZE - 20, 2)
    r = int(rng.integers(6, 18))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=220)
    return np.asarray(pil, dtype=float)


def _add_tread_uneven(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    period = int(rng.integers(8, 20))
    amp = float(rng.uniform(10, 25))
    phase = float(rng.uniform(0, 2 * np.pi))
    return img + amp * np.sin(2 * np.pi * x / period + phase)


def _add_layer_separation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    x1, y1 = rng.integers(0, IMAGE_SIZE, 2)
    x2, y2 = rng.integers(0, IMAGE_SIZE, 2)
    draw.line([int(x1), int(y1), int(x2), int(y2)],
              fill=25, width=int(rng.integers(2, 6)))
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


def generate_dataset(output_dir, n_per_class: int = 50, seed: int = 42) -> pd.DataFrame:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows = []

    for defect_type in CLASSES:
        for i in range(n_per_class):
            tire_id = f"TIRE-{defect_type[:3].upper()}-{i:04d}"
            img = make_image(defect_type, rng)
            Image.fromarray(img).save(images_dir / f"{tire_id}.png")

            rows.append({
                "tire_id": tire_id,
                "batch_id": f"B{int(rng.integers(1, 9)):03d}",
                "machine_id": f"M{int(rng.integers(1, 6)):02d}",
                "mold_id": f"D{int(rng.integers(1, 9)):02d}",
                "material_lot": f"L{int(rng.integers(1, 7)):02d}",
                "build_temp": float(rng.uniform(140, 170)),
                "build_pressure": float(rng.uniform(8, 14)),
                "cure_temp": float(rng.uniform(150, 180)),
                "cure_time": float(rng.uniform(10, 20)),
                "defect_type": defect_type,
                "defect_probability": float(
                    0.05 if defect_type == "good" else rng.uniform(0.55, 0.98)
                ),
                "severity": (
                    "high" if defect_type == "layer_separation"
                    else "medium" if defect_type in ("bubble", "tread_uneven")
                    else "low"
                ),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "metadata.csv", index=False)
    return df


# =============================================================================
# 6) SERVICES
# =============================================================================

class VisionService:
    def __init__(self, model_path=MODEL_PATH):
        self.model = DefectModel.load(model_path)

    def predict_bytes(self, data: bytes) -> dict:
        img = Image.open(io.BytesIO(data)).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        features = extract_features(img)
        result = self.model.predict(features)

        p = result["defect_probability"]
        severity = "high" if p >= 0.8 else "medium" if p >= 0.5 else "low"
        return {**result, "severity": severity, "features": features}

    def predict_path(self, path) -> dict:
        return self.predict_bytes(Path(path).read_bytes())


class TraceabilityService:
    def __init__(self, metadata_path=METADATA_PATH):
        path = Path(metadata_path)
        self.df = pd.read_csv(path) if path.exists() else None

    def trace(self, metadata: dict, defect_type: str, top_k: int = 3):
        if self.df is None or self.df.empty:
            return []

        df = self.df
        baseline = max(float((df["defect_type"] == defect_type).mean()), 1e-6)
        causes = []

        for field in ["machine_id", "mold_id", "material_lot", "batch_id"]:
            value = metadata.get(field)
            if value is None:
                continue
            subset = df[df[field].astype(str) == str(value)]
            if subset.empty:
                continue

            rate = float((subset["defect_type"] == defect_type).mean())
            lift = float(rate / baseline)
            causes.append({
                "field": field,
                "value": str(value),
                "defect_rate": rate,
                "lift": lift,
            })

        causes.sort(key=lambda x: x["defect_rate"] * x["lift"], reverse=True)
        return causes[:top_k]


class RiskService:
    def predict(self, metadata: dict, defect_probability: float):
        risk = float(defect_probability)
        reasons = []

        if metadata.get("build_temp", 0) > 168:
            risk += 0.08
            reasons.append("دمای ساخت بالاتر از حد بهینه")
        if metadata.get("build_pressure", 0) < 9.5:
            risk += 0.08
            reasons.append("فشار ساخت پایین")
        if metadata.get("cure_temp", 0) < 155 or metadata.get("cure_temp", 0) > 175:
            risk += 0.10
            reasons.append("دمای پخت خارج از بازه بهینه")
        if metadata.get("cure_time", 0) < 12:
            risk += 0.07
            reasons.append("زمان پخت کوتاه")

        risk = min(0.99, max(0.01, risk))
        level = "high" if risk >= 0.75 else "medium" if risk >= 0.45 else "low"
        return risk, level, reasons


def grade_decision(defect_type: str, defect_probability: float, severity: str):
    if defect_type == "good" and defect_probability < 0.35:
        return "A", "پذیرش", ["احتمال عیب پایین"]

    if defect_type == "layer_separation" or defect_probability >= 0.90:
        return "C", "ضایعات/بازکاری با بازبینی اجباری", [
            "عیب بحرانی یا احتمال بسیار بالا"
        ]

    if defect_probability >= 0.65 or severity == "high":
        return "B", "بازکاری/بازبینی", [
            "نیاز به بازبینی انسانی و اصلاح فرآیند"
        ]

    return "A", "پذیرش با پایش", ["عیب مرزی با احتمال متوسط"]


# =============================================================================
# 7) API (FastAPI)
# =============================================================================

app = FastAPI(title="TVDI API", version="0.1.0")

_vision_service = VisionService()
_trace_service = TraceabilityService()
_risk_service = RiskService()


def _run_prediction(image_bytes: bytes, metadata: TireMetadata) -> PredictResponse:
    meta = metadata.model_dump()
    vision = _vision_service.predict_bytes(image_bytes)

    causes = _trace_service.trace(meta, vision["defect_type"])
    risk_score, risk_level, risk_reasons = _risk_service.predict(
        meta, vision["defect_probability"]
    )
    grade_value, recommendation, grade_reasons = grade_decision(
        vision["defect_type"],
        vision["defect_probability"],
        vision["severity"],
    )

    return PredictResponse(
        tire_id=metadata.tire_id,
        defect_type=vision["defect_type"],
        defect_probability=vision["defect_probability"],
        severity=vision["severity"],
        suspected_causes=[Cause(**c) for c in causes],
        risk_score=risk_score,
        risk_level=risk_level,
        grade=grade_value,
        recommendation=recommendation,
        reasons=grade_reasons + risk_reasons,
        features=vision["features"],
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "TVDI"}


@app.post("/predict-json", response_model=PredictResponse)
def predict_json(request: PredictRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc
    return _run_prediction(image_bytes, request.metadata)


@app.post("/predict", response_model=PredictResponse)
async def predict_multipart(
    file: UploadFile = File(...),
    meta: str = Form(...),
):
    try:
        metadata = TireMetadata(**json.loads(meta))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid meta json: {exc}") from exc
    return _run_prediction(await file.read(), metadata)


# =============================================================================
# 8) TRAINING
# =============================================================================

def train(data_dir=DATA_DIR, model_path=MODEL_PATH):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "metadata.csv")

    X, y = [], []
    for _, row in df.iterrows():
        img_path = data_dir / "images" / f"{row['tire_id']}.png"
        img = Image.open(img_path).convert("L")
        X.append(features_to_vector(extract_features(img)))
        y.append(row["defect_type"])

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred, zero_division=0))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"model saved to {model_path}")


# =============================================================================
# 9) TESTS (pytest-compatible)
# =============================================================================

def test_generate_dataset(tmp_path=None):
    tmp_path = Path(tmp_path or ROOT_DIR / "data_test")
    df = generate_dataset(tmp_path, n_per_class=2, seed=1)

    assert len(df) == len(CLASSES) * 2
    assert (tmp_path / "metadata.csv").exists()
    assert len(list((tmp_path / "images").glob("*.png"))) == len(CLASSES) * 2
    assert set(df["defect_type"]) == set(CLASSES)


def test_feature_extraction():
    img = np.zeros((128, 128), dtype=np.uint8)
    features = extract_features(img)

    assert set(features.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features.values())


def test_heuristic_bubble():
    features = {
        "mean": 0, "std": 0, "min": 0, "max": 0,
        "p10": 0, "p50": 0, "p90": 0,
        "dark_ratio": 0, "bright_ratio": 0.05,
        "grad_mean": 0, "grad_std": 0,
        "lap_var": 500, "edge_density": 0,
    }
    result = DefectModel().predict(features)

    assert result["defect_type"] == "bubble"
    assert result["defect_probability"] > 0.5


def test_api_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_predict_json():
    buf = io.BytesIO()
    Image.new("L", (128, 128), 128).save(buf, format="PNG")
    payload = {
        "image_base64": base64.b64encode(buf.getvalue()).decode(),
        "metadata": {
            "tire_id": "T1", "batch_id": "B1",
            "machine_id": "M01", "mold_id": "D01",
            "material_lot": "L01",
            "build_temp": 155, "build_pressure": 11,
            "cure_temp": 165, "cure_time": 15,
        },
    }
    client = TestClient(app)
    response = client.post("/predict-json", json=payload)
    assert response.status_code == 200
    body = response.json()
    for key in ("defect_type", "defect_probability", "grade", "recommendation"):
        assert key in body


# =============================================================================
# 10) CLI ENTRYPOINT
# =============================================================================

def _cmd_generate(args):
    df = generate_dataset(args.out, n_per_class=args.n, seed=args.seed)
    print(f"generated {len(df)} samples in {args.out}")


def _cmd_train(args):
    train(args.data, args.model)


def _cmd_serve(args):
    import uvicorn
    uvicorn.run("src:app", host=args.host, port=args.port, reload=args.reload)


def _cmd_test(_args):
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVDI single-file toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="تولید داده مصنوعی")
    p_gen.add_argument("--n", type=int, default=50)
    p_gen.add_argument("--out", type=str, default="data")
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.set_defaults(func=_cmd_generate)

    p_train = sub.add_parser("train", help="آموزش مدل")
    p_train.add_argument("--data", type=str, default="data")
    p_train.add_argument("--model", type=str, default="models/defect_classifier.joblib")
    p_train.set_defaults(func=_cmd_train)

    p_serve = sub.add_parser("serve", help="اجرای سرور FastAPI")
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    p_test = sub.add_parser("test", help="اجرای تست‌ها")
    p_test.set_defaults(func=_cmd_test)

    args = parser.parse_args()
    args.func(args)
