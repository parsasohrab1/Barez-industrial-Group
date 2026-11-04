"""MQTT Client با Sparkplug B
- انتشار رویداد عیب (NDATA/DDATA)
- دریافت فرمان از MES
- Store-and-forward برای قطع شبکه
"""
import json
import logging
import time
import uuid
from typing import Callable, Optional

from tvdi_industrial.config import get_settings

logger = logging.getLogger(__name__)


class MQTTClient:
    def __init__(self, settings=None, client_id: Optional[str] = None):
        self.s = settings or get_settings()
        self.client_id = client_id or self.s.mqtt_client_id
        self._client = None
        self._mqtt = None
        self._handlers: dict[str, Callable] = {}
        self._connected = False

    def connect(self):
        if not self.s.mqtt_host:
            logger.info("MQTT host is empty; store-and-forward only")
            return
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.warning("paho-mqtt is not installed")
            return
        self._mqtt = mqtt
        protocol = mqtt.MQTTv311 if str(self.s.mqtt_protocol).startswith("3") else mqtt.MQTTv5
        kwargs = {"client_id": self.client_id, "protocol": protocol}
        if hasattr(mqtt, "CallbackAPIVersion"):
            kwargs["callback_api_version"] = mqtt.CallbackAPIVersion.VERSION1
        self._client = mqtt.Client(**kwargs)
        if self.s.mqtt_username:
            self._client.username_pw_set(self.s.mqtt_username, self.s.mqtt_password)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        try:
            self._client.connect_async(self.s.mqtt_host, self.s.mqtt_port, keepalive=30)
            self._client.loop_start()
        except Exception as exc:
            logger.warning("MQTT connect failed: %s", exc)
            self._client = None

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        code = getattr(rc, "value", rc)
        self._connected = code == 0 or code == "Success"
        logger.info("MQTT connected rc=%s", rc)
        for topic in self._handlers:
            client.subscribe(topic, qos=self.s.mqtt_qos)

    def _on_disconnect(self, client, userdata, rc, properties=None):
        self._connected = False
        logger.warning("MQTT disconnected rc=%s", rc)

    @property
    def connected(self) -> bool:
        return self._connected

    def wait_connected(self, timeout: float = 3.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._connected:
                return True
            time.sleep(0.05)
        return False

    def publish_defect_event(self, tire_id: str, payload: dict) -> bool:
        """Publish a defect event. Returns False when the broker is unavailable."""
        topic = f"{self.s.mqtt_topic_prefix}/DDATA/{self.client_id}/{tire_id}"
        body = {
            "timestamp": int(time.time() * 1000),
            "uuid": str(uuid.uuid4()),
            **payload,
        }
        if self._client is None or not self._connected or self._mqtt is None:
            logger.warning("MQTT unavailable; caller should buffer %s", topic)
            return False
        info = self._client.publish(topic, json.dumps(body), qos=self.s.mqtt_qos)
        if info.rc != self._mqtt.MQTT_ERR_SUCCESS:
            logger.error("publish failed rc=%s", info.rc)
            return False
        return True

    def publish_node_birth(self) -> None:
        if self._client is None:
            return
        topic = f"{self.s.mqtt_topic_prefix}/NBIRTH/{self.client_id}"
        self._client.publish(topic, json.dumps({
            "timestamp": int(time.time() * 1000),
            "bdSeq": 0,
        }), qos=self.s.mqtt_qos)

    def publish_node_death(self) -> None:
        if self._client is None:
            return
        topic = f"{self.s.mqtt_topic_prefix}/NDEATH/{self.client_id}"
        self._client.publish(topic, json.dumps({"bdSeq": 0}), qos=self.s.mqtt_qos)

    def subscribe(self, topic: str, handler: Callable[[str, dict], None]):
        """اشتراک در MES/SCADA commands"""
        self._handlers[topic] = handler
        if self._connected and self._client is not None:
            self._client.subscribe(topic, qos=self.s.mqtt_qos)
            self._client.on_message = self._on_message

    def _on_message(self, client, userdata, msg):
        handler = self._handlers.get(msg.topic)
        if not handler:
            return
        try:
            payload = json.loads(msg.payload)
            handler(msg.topic, payload)
        except Exception as exc:
            logger.error("handler error: %s", exc)

    def disconnect(self):
        if self._client is None:
            return
        self.publish_node_death()
        self._client.loop_stop()
        self._client.disconnect()
