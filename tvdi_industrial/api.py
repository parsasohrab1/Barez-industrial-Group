"""Single TVDI API used by the dashboard, the CLI, and Docker."""
import asyncio
import io
import logging
import json
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from tvdi_industrial.config import get_settings
from tvdi_industrial.cycle import software_cycle
from tvdi_industrial.decision import (
    LAB_LIMITS,
    RiskService,
    channel_report,
    factory_grade,
    grade_decision,
    severity_of,
    station_decision,
)
from tvdi_industrial.defect_model import DefectModel
from tvdi_industrial.edge.buffer import EdgeBuffer
from tvdi_industrial.edge.inference import ONNXInference
from tvdi_industrial.features import IMAGE_SIZE, extract_features
from tvdi_industrial.industrial.lab_line import LabLine
from tvdi_industrial.industrial.mqtt_client import MQTTClient
from tvdi_industrial.industrial.opcua_client import OPCUAClient
from tvdi_industrial.industrial.reject_actuator import RejectActuator
from tvdi_industrial.notifications.service import AlertSeverity, NotificationService
from tvdi_industrial.observability.health import health, router as health_router
from tvdi_industrial.observability.logging import get_logger, new_correlation_id, setup_logging
from tvdi_industrial.observability.metrics import (
    MODEL_LOADED,
    MQTT_BUFFER,
    OPCUA_UP,
    PREDICT_LATENCY,
    PREDICTIONS,
    REJECTS,
)
from tvdi_industrial.persistence.repository import Repository
from tvdi_industrial.security.audit import AuditLog
from tvdi_industrial.security.auth import (
    AuthService,
    Role,
    TokenUser,
    require_permission,
)

setup_logging()
log = get_logger("api")
logging.getLogger("asyncua").setLevel(logging.WARNING)
_risk = RiskService()
DEV_CREDENTIALS = Path("data/dev-credentials.json")


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
    auto_reject: bool = False


class PredictRequest(BaseModel):
    image_base64: str
    metadata: TireMetadata = Field(default_factory=TireMetadata)


class LoginRequest(BaseModel):
    username: str
    password: str


def _load_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(image)


def _store_image(tire_id: str, image_bytes: bytes) -> str:
    folder = Path("data/inspections")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{tire_id}-{int(time.time() * 1000)}.png"
    path.write_bytes(image_bytes)
    return str(path)


def _ensure_dev_users(auth: AuthService, settings) -> None:
    if settings.tvdi_env == "production":
        if settings.jwt_secret.startswith("change-me"):
            raise RuntimeError("Set JWT_SECRET before running with TVDI_ENV=production")
        return
    DEV_CREDENTIALS.parent.mkdir(parents=True, exist_ok=True)
    if DEV_CREDENTIALS.exists():
        creds = json.loads(DEV_CREDENTIALS.read_text(encoding="utf-8"))
    else:
        creds = {"username": "operator", "password": secrets.token_urlsafe(9)}
        DEV_CREDENTIALS.write_text(json.dumps(creds, indent=2), encoding="utf-8")
    auth.register_user(creds["username"], creds["password"], [Role.OPERATOR, Role.QUALITY_MANAGER])
    log.warning("development login file: %s", DEV_CREDENTIALS)


async def _open_repository(settings) -> Repository:
    repo = Repository(settings.database_url)
    try:
        await repo.init_db()
        return repo
    except Exception:
        log.exception("primary database failed; falling back to local sqlite")
        fallback = Repository("sqlite+aiosqlite:///./data/tvdi.db")
        await fallback.init_db()
        return fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.lab_line = None
    if settings.lab_line_enabled:
        app.state.lab_line = await _start_lab_line(settings)
    app.state.inference = ONNXInference()
    app.state.inference.load()
    app.state.joblib = DefectModel.load()
    app.state.repo = await _open_repository(settings)
    app.state.opcua = OPCUAClient()
    app.state.mqtt = MQTTClient()
    app.state.mqtt.connect()
    if settings.lab_line_enabled:
        await asyncio.to_thread(app.state.mqtt.wait_connected, 3)
    app.state.mqtt.publish_node_birth()
    app.state.buffer = EdgeBuffer()
    app.state.reject = RejectActuator(app.state.opcua)
    app.state.auth = AuthService()
    _ensure_dev_users(app.state.auth, settings)
    app.state.audit = AuditLog()
    app.state.notify = NotificationService()

    if settings.opcua_endpoint:
        try:
            await asyncio.wait_for(app.state.opcua.connect(), timeout=8)
        except Exception as exc:
            log.warning("OPC UA not connected: %s", exc)

    def inference_state():
        if app.state.inference._session is not None:
            return "onnx"
        if app.state.joblib.model is not None:
            return "joblib"
        return "heuristic"

    def line_check(up: bool):
        if not up:
            return False
        return "lab" if settings.lab_line_enabled else True

    health.register("database", lambda: app.state.repo.ready is True)
    health.register("inference", inference_state)
    health.register("opcua", lambda: line_check(app.state.opcua._client is not None))
    health.register("mqtt", lambda: line_check(bool(app.state.mqtt.connected)))
    _refresh_gauges(app)
    log.info("startup complete")
    yield
    app.state.mqtt.disconnect()
    if app.state.opcua._client is not None:
        await app.state.opcua.close()
    if app.state.lab_line is not None:
        await app.state.lab_line.stop()
    log.info("shutdown complete")


async def _start_lab_line(settings):
    errors = []
    for mqtt_port, opcua_port in ((1883, 4840), (1884, 4841)):
        lab = LabLine(mqtt_port=mqtt_port, opcua_port=opcua_port)
        try:
            await lab.start()
        except Exception as exc:
            errors.append(f"{mqtt_port}/{opcua_port}: {exc}")
            await lab.stop()
            continue
        settings.opcua_endpoint = lab.opcua_endpoint
        settings.opcua_namespace_index = lab.namespace_index
        settings.mqtt_host = lab.mqtt_host
        settings.mqtt_port = lab.mqtt_port
        settings.mqtt_protocol = "3.1.1"
        return lab
    log.warning("lab line did not start: %s", "; ".join(errors))
    return None


def _refresh_gauges(app: FastAPI) -> None:
    MODEL_LOADED.set(1 if app.state.inference._session is not None else 0)
    OPCUA_UP.set(1 if app.state.opcua._client is not None else 0)
    try:
        MQTT_BUFFER.set(len(app.state.buffer.pending()))
    except Exception:
        MQTT_BUFFER.set(0)


app = FastAPI(title="TVDI API", version="0.2.0", lifespan=lifespan)
app.include_router(health_router)


@app.get("/real/report")
async def real_report():
    path = Path("data/real/trl5-report.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="real X-ray report has not been generated")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/predict-xray")
async def predict_xray(file: UploadFile = File(...), user: TokenUser = Depends(require_permission("predict:read"))):
    from tvdi_industrial.real_xray import MODEL_PATH, predict_xray as score_xray

    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="real X-ray model has not been trained")
    return score_xray(await file.read())


@app.get("/real/localization")
async def localization_report():
    path = Path("data/real/localization-report.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="localization report has not been generated")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/lab/cycle")
async def cycle_report():
    path = Path("data/lab/cycle-report.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="cycle report has not been generated")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/lab/report")
async def lab_report():
    path = Path("data/lab/trl4-report.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="laboratory report has not been generated")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
async def health_root():
    data = await health.run()
    checks = data.get("checks", {})
    inference = str(checks.get("inference", "fail"))
    servable = checks.get("database") == "ok" and not inference.startswith("fail")
    return JSONResponse(
        {
            "status": "ok" if servable else "down",
            "service": "TVDI",
            "readiness": data["status"],
            "checks": checks,
        },
        status_code=200 if servable else 503,
    )


@app.post("/auth/login")
async def login(body: LoginRequest):
    import jwt

    token_user = app.state.auth.authenticate(body.username, body.password)
    if not token_user:
        raise HTTPException(status_code=401, detail="invalid credentials")
    settings = get_settings()
    payload = {
        "sub": token_user.sub,
        "roles": [role.value for role in token_user.roles],
        "exp": token_user.exp,
    }
    return {
        "access_token": jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm),
        "token_type": "bearer",
    }


async def _inspect(image_bytes: bytes, metadata: TireMetadata, user: TokenUser) -> dict:
    started = time.time()
    new_correlation_id()
    array = _load_image(image_bytes)
    features = extract_features(array)
    vision = app.state.inference.predict(array)
    if vision["source"] == "heuristic":
        trained = app.state.joblib.predict(features)
        if trained["source"] == "joblib":
            vision = trained
    vision["severity"] = severity_of(vision["defect_probability"])
    meta = metadata.model_dump()

    line = {}
    line_source = None
    if app.state.settings.opcua_endpoint and app.state.opcua._client is not None:
        try:
            line = await app.state.opcua.read_line_data()
            for key, value in line.items():
                if value is not None:
                    meta[key] = value
            line_source = "lab_fixture" if app.state.settings.lab_line_enabled else "opcua"
        except Exception as exc:
            log.warning("OPC UA process read failed: %s", exc)

    causes = await app.state.repo.trace(meta, vision["defect_type"])
    risk_score, risk_level, risk_reasons = _risk.predict(meta, vision["defect_probability"])
    grade, recommendation, grade_reasons = grade_decision(
        vision["defect_type"], vision["defect_probability"], vision["severity"],
    )
    limits = LAB_LIMITS if line_source == "lab_fixture" else None
    factory = factory_grade(line, limits)
    station = station_decision(grade, factory)
    timing = software_cycle(array, lambda canvas: app.state.joblib.predict(extract_features(np.asarray(canvas))))
    result = {
        "tire_id": metadata.tire_id,
        "defect_type": vision["defect_type"],
        "defect_probability": vision["defect_probability"],
        "severity": vision["severity"],
        "suspected_causes": causes,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "grade": grade,
        "recommendation": recommendation,
        "reasons": grade_reasons + risk_reasons,
        "source": vision["source"],
        "features": features,
        "channels": channel_report(vision["defect_type"]),
        "factory_grade": factory,
        "station": station,
        "timing": timing,
        "line_source": line_source,
    }
    result["image_path"] = _store_image(metadata.tire_id, image_bytes)
    await app.state.repo.save_inspection(meta, result, app.state.settings.model_version)

    published = app.state.mqtt.publish_defect_event(metadata.tire_id, {
        "defect_type": result["defect_type"],
        "grade": result["grade"],
        "risk_score": result["risk_score"],
    })
    if not published:
        topic = f"{app.state.settings.mqtt_topic_prefix}/DDATA/{app.state.mqtt.client_id}/{metadata.tire_id}"
        app.state.buffer.enqueue(topic, result)
    _refresh_gauges(app)

    app.state.audit.log(user.sub, "predict", metadata.tire_id, {
        "grade": grade, "defect_type": result["defect_type"], "source": result["source"],
    })
    PREDICTIONS.labels(defect_type=result["defect_type"], grade=grade).inc()
    PREDICT_LATENCY.observe(time.time() - started)

    if risk_level == "high":
        try:
            await app.state.notify.send(
                AlertSeverity.CRITICAL,
                f"عیب بحرانی {metadata.tire_id}",
                f"نوع: {result['defect_type']} — ریسک {risk_score * 100:.0f}%",
            )
        except Exception as exc:
            log.warning("notification failed: %s", exc)

    if metadata.auto_reject and grade == "C":
        if not app.state.settings.reject_enabled:
            result["reject"] = {"tire_id": metadata.tire_id, "status": "disabled", "reason": result["defect_type"]}
        else:
            reject_result = await app.state.reject.reject(metadata.tire_id, result["defect_type"])
            REJECTS.labels(status=reject_result["status"]).inc()
            result["reject"] = reject_result
            app.state.audit.log(user.sub, "reject", metadata.tire_id, reject_result)
    return result


@app.post("/predict-json")
async def predict_json(body: PredictRequest, user: TokenUser = Depends(require_permission("predict:read"))):
    import base64

    try:
        image_bytes = base64.b64decode(body.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc
    return await _inspect(image_bytes, body.metadata, user)


@app.post("/predict")
async def predict_multipart(
    file: UploadFile = File(...),
    meta: str = Form(...),
    user: TokenUser = Depends(require_permission("predict:read")),
):
    try:
        metadata = TireMetadata(**json.loads(meta))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid meta json: {exc}") from exc
    return await _inspect(await file.read(), metadata, user)


@app.get("/inspections")
async def inspections(hours: int = 48, user: TokenUser = Depends(require_permission("predict:read"))):
    return await app.state.repo.list_inspections(hours=hours)


@app.get("/stats/defect-rate")
async def stats(hours: int = 24, user: TokenUser = Depends(require_permission("predict:read"))):
    return await app.state.repo.defect_rate(hours=hours)


@app.post("/reject/{tire_id}")
async def reject(tire_id: str, user: TokenUser = Depends(require_permission("inspect:create"))):
    if not app.state.settings.reject_enabled:
        return {"tire_id": tire_id, "status": "disabled"}
    result = await app.state.reject.reject(tire_id, "manual")
    app.state.audit.log(user.sub, "reject", tire_id, result)
    REJECTS.labels(status=result["status"]).inc()
    return result


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
