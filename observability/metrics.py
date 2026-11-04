"""متریک‌های Prometheus برای API و مدل"""
from prometheus_client import Counter, Histogram, Gauge

PREDICTIONS = Counter(
    "tvdi_predictions_total", "Total predictions",
    ["defect_type", "grade"],
)
PREDICT_LATENCY = Histogram(
    "tvdi_predict_latency_seconds", "Prediction latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5],
)
REJECTS = Counter("tvdi_rejects_total", "Reject commands", ["status"])
MQTT_BUFFER = Gauge("tvdi_mqtt_buffer_size", "Pending MQTT events")
OPCUA_UP = Gauge("tvdi_opcua_up", "OPC UA connection status")
MODEL_LOADED = Gauge("tvdi_model_loaded", "1 if ONNX model loaded")
DRIFT_SCORE = Gauge("tvdi_drift_score", "Feature drift score")
