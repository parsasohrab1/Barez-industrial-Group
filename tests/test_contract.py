import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tvdi_industrial.config import get_settings
from tvdi_industrial.observability.health import HealthRegistry
from tvdi_industrial.synthetic import make_image


@pytest.mark.asyncio
async def test_health_false_is_a_failure():
    registry = HealthRegistry()
    registry.register("database", lambda: True)
    registry.register("opcua", lambda: False)
    data = await registry.run()
    assert data["checks"]["opcua"] == "fail"
    assert data["checks"]["database"] == "ok"
    assert data["status"] == "degraded"


def test_api_saves_inspection_and_keeps_reject_off(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path.as_posix()}/tvdi.db")
    monkeypatch.setenv("MQTT_HOST", "")
    monkeypatch.setenv("OPCUA_ENDPOINT", "")
    monkeypatch.setenv("TVDI_ENV", "development")
    monkeypatch.setenv("REJECT_ENABLED", "false")
    get_settings.cache_clear()

    from tvdi_industrial.api import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["checks"]["inference"] in {"heuristic", "joblib"}
        assert health.json()["checks"]["opcua"] == "fail"

        creds = json.loads(Path("data/dev-credentials.json").read_text(encoding="utf-8"))
        login = client.post("/auth/login", json=creds)
        assert login.status_code == 200
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        image = io.BytesIO()
        specimen = make_image("layer_separation", np.random.default_rng(1))
        Image.fromarray(specimen).save(image, format="PNG")
        denied = client.post("/predict-json", json={"image_base64": "aa", "metadata": {"tire_id": "X"}})
        assert denied.status_code == 401

        response = client.post(
            "/predict-json",
            headers=headers,
            json={
                "image_base64": base64.b64encode(image.getvalue()).decode(),
                "metadata": {"tire_id": "T-CONTRACT", "auto_reject": True},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["source"] in {"heuristic", "joblib"}
        assert body["grade"] == "C"
        assert body["reject"]["status"] == "disabled"
        assert isinstance(body["suspected_causes"], list)
        assert body["channels"]["shearography"]["status"] == "unavailable"
        assert body["channels"]["appearance"]["coordinates"] is None
        assert body["factory_grade"]["status"] == "incomplete"
        assert body["station"]["status"] == "incomplete"
        assert body["timing"]["motion_measured"] is False
        assert body["timing"]["sectors"] == 16

        listed = client.get("/inspections", headers=headers)
        assert listed.status_code == 200
        assert any(item["tire_id"] == "T-CONTRACT" for item in listed.json())
