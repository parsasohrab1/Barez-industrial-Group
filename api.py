"""API یکپارچه که همه سرویس‌ها را به هم می‌زند"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from tvdi_industrial.config import get_settings
from tvdi_industrial.edge.inference import ONNXInference
from tvdi_industrial.observability.health import health, router as health_router
from tvdi_industrial.observability.logging import setup_logging, new_correlation_id, get_logger
from tvdi_industrial.observability.metrics import PREDICTIONS, PREDICT_LATENCY, REJECTS
from tvdi_industrial.persistence.repository import Repository
from tvdi_industrial.security.audit import AuditLog
from tvdi_industrial.security.auth import (
    AuthService, Role, get_current_user, require_permission, TokenUser,
)
from tvdi_industrial.industrial.opcua_client import OPCUAClient
from tvdi_industrial.industrial.mqtt_client import MQTTClient
from tvdi_industrial.industrial.reject_actuator import RejectActuator
from tvdi_industrial.notifications.service import NotificationService, AlertSeverity

setup_logging()
log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.inference = ONNXInference()
    app.state.inference.load()
    app.state.repo = Repository()
    await app.state.repo.init_db()
    app.state.opcua = OPCUAClient()
    app.state.mqtt = MQTTClient()
    app.state.mqtt.connect()
    app.state.mqtt.publish_node_birth()
    app.state.reject = RejectActuator(app.state.opcua)
    app.state.auth = AuthService()
    app.state.auth.register_user("admin", "admin", [Role.ADMIN])
    app.state.auth.register_user("operator", "operator", [Role.OPERATOR])
    app.state.audit = AuditLog()
    app.state.notify = NotificationService()

    health.register("model", lambda: app.state.inference._session is not None)
    health.register("opcua", lambda: app.state.opcua._client is not None)
    health.register("mqtt", lambda: app.state.mqtt.connected)
    log.info("startup complete")
    yield

    app.state.mqtt.disconnect()
    await app.state.opcua.close()
    log.info("shutdown complete")


app = FastAPI(title="TVDI Industrial API", version="1.0.0", lifespan=lifespan)
app.include_router(health_router)


# ── Schemas ──────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictMeta(BaseModel):
    tire_id: str
    batch_id: str = "B0001"
    machine_id: str = "M01"
    mold_id: str = "D01"
    material_lot: str = "L01"
    build_temp: float = 155.0
    build_pressure: float = 11.0
    cure_temp: float = 165.0
    cure_time: float = 15.0
    auto_reject: bool = False


# ── Auth ─────────────────────────────────────────────────────────────
@app.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    token = app.state.auth.authenticate(req.username, req.password)
    if not token:
        raise HTTPException(status_code=401, detail="invalid credentials")
    import jwt
    s = get_settings()
    payload = {"sub": token.sub, "roles": [r.value for r in token.roles], "exp": token.exp}
    return LoginResponse(
        access_token=jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)
    )


# ── Predict ──────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    meta: str = Form(...),
    user: TokenUser = Depends(require_permission("predict:read")),
):
    import json as _json
    import time
    from PIL import Image
    import io

    new_correlation_id()
    t0 = time.time()

    try:
        metadata = PredictMeta(**_json.loads(meta))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"meta invalid: {exc}")

    img_bytes = await file.read()
    arr = __import__("numpy").array(Image.open(io.BytesIO(img_bytes)).convert("L"))

    result = app.state.inference.predict(arr)
    result["risk_score"] = min(0.99, result["defect_probability"] * 1.1)
    result["risk_level"] = "high" if result["risk_score"] > 0.75 else \
                            "medium" if result["risk_score"] > 0.45 else "low"
    result["grade"] = "C" if result["defect_type"] == "layer_separation" else \
                      "B" if result["defect_probability"] > 0.65 else "A"
    result["recommendation"] = "ضایعات" if result["grade"] == "C" else \
                                "بازبینی" if result["grade"] == "B" else "پذیرش"
    result["severity"] = "high" if result["defect_probability"] > 0.8 else \
                        "medium" if result["defect_probability"] > 0.5 else "low"

    await app.state.repo.save_inspection(metadata.model_dump(), result,
                                          get_settings().model_version)
    app.state.mqtt.publish_defect_event(metadata.tire_id, result)

    PREDICTIONS.labels(defect_type=result["defect_type"], grade=result["grade"]).inc()
    PREDICT_LATENCY.observe(time.time() - t0)
    app.state.audit.log(user.sub, "predict", metadata.tire_id, result)

    if result["risk_level"] == "high":
        await app.state.notify.send(
            AlertSeverity.CRITICAL,
            f"عیب بحرانی {metadata.tire_id}",
            f"نوع: {result['defect_type']} — ریسک {result['risk_score']*100:.0f}%",
        )

    if metadata.auto_reject and result["grade"] == "C":
        reject_result = await app.state.reject.reject(metadata.tire_id, result["defect_type"])
        REJECTS.labels(status=reject_result["status"]).inc()
        result["reject"] = reject_result

    return result


# ── Reject (manual) ──────────────────────────────────────────────────
@app.post("/reject/{tire_id}")
async def reject(
    tire_id: str,
    user: TokenUser = Depends(require_permission("inspect:create")),
):
    result = await app.state.reject.reject(tire_id, "manual")
    app.state.audit.log(user.sub, "reject", tire_id, result)
    REJECTS.labels(status=result["status"]).inc()
    return result


# ── Metrics ──────────────────────────────────────────────────────────
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Stats ────────────────────────────────────────────────────────────
@app.get("/stats/defect-rate")
async def stats(user: TokenUser = Depends(require_permission("report:export"))):
    return await app.state.repo.defect_rate(hours=24)
