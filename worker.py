"""Worker پس‌زمینه برای همگام‌سازی buffer آفلاین با MQTT"""
import asyncio
import json
import logging

from tvdi_industrial.config import get_settings
from tvdi_industrial.edge.buffer import EdgeBuffer
from tvdi_industrial.industrial.mqtt_client import MQTTClient

logger = logging.getLogger(__name__)


async def run():
    s = get_settings()
    buffer = EdgeBuffer()
    mqtt = MQTTClient()
    mqtt.connect()

    async def loop():
        while True:
            if mqtt.connected:
                def sender(topic, payload):
                    try:
                        mqtt._client.publish(topic, json.dumps(payload), qos=s.mqtt_qos)
                        return True
                    except Exception as exc:
                        logger.error("sync send failed: %s", exc)
                        return False
                buffer.sync(sender)
            await asyncio.sleep(s.edge_sync_interval_s)

    await loop()


if __name__ == "__main__":
    asyncio.run(run())
