"""پیکربندی مرکزی از متغیرهای محیطی"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── OPC UA ─────────────────────────────────────────
    opcua_endpoint: str = "opc.tcp://plc.local:4840"
    opcua_username: str = ""
    opcua_password: str = ""
    opcua_namespace: str = "urn:barez:tvdi"
    opcua_reconnect_s: int = 5

    # ── MQTT ───────────────────────────────────────────
    mqtt_host: str = "mosquitto"
    mqtt_port: int = 1883
    mqtt_client_id: str = "tvdi-edge-01"
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_qos: int = 1
    mqtt_topic_prefix: str = "spBv1.0/BAREZ"

    # ── Modbus ─────────────────────────────────────────
    modbus_host: str = "plc-legacy.local"
    modbus_port: int = 502
    modbus_unit_id: int = 1

    # ── Database ───────────────────────────────────────
    database_url: str = "postgresql+asyncpg://tvdi:tvdi@postgres:5432/tvdi"

    # ── Auth ───────────────────────────────────────────
    jwt_secret: str = "change-me-in-prod-use-vault"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    oidc_discovery_url: str = ""

    # ── Model ──────────────────────────────────────────
    model_path: str = "models/defect.onnx"
    model_fallback_path: str = "models/defect_classifier.joblib"
    model_registry_dir: str = "models/registry"
    model_version: str = "v1.0.0"

    # ── Notifications ──────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    sms_api_key: str = ""

    # ── Edge ───────────────────────────────────────────
    edge_buffer_db: str = "edge_buffer.sqlite"
    edge_sync_interval_s: int = 10

    # ── Reject Actuator ────────────────────────────────
    reject_plc_confirm_timeout_ms: int = 500
    reject_enabled: bool = True

    # ── Observability ──────────────────────────────────
    log_level: str = "INFO"
    service_name: str = "tvdi"


@lru_cache
def get_settings() -> Settings:
    return Settings()
