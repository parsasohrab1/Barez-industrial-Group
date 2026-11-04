"""In-process OPC UA server and MQTT broker.

Values written here are a laboratory fixture. They are not Barez process tags.
"""
import asyncio
import logging

from tvdi_industrial.industrial.lab_broker import LabMqttBroker
from tvdi_industrial.industrial.opcua_client import OPCUAServerStub

logger = logging.getLogger(__name__)

LAB_VALUES = {
    "Build.Temp": 160.0,
    "Build.Pressure": 10.5,
    "Cure.Temp": 165.0,
    "Cure.Time": 14.0,
    "Uniformity.RFV": 62.0,
    "Uniformity.LFV": 41.0,
    "Balance.Static": 18.0,
    "Balance.Couple": 320.0,
    "Laser.Bulge": 0.35,
    "Laser.Depression": 0.28,
    "Laser.RadialRunout": 0.55,
    "Reject.Trigger": False,
}


class LabLine:
    def __init__(self, mqtt_port: int = 1883, opcua_port: int = 4840):
        self.mqtt_host = "127.0.0.1"
        self.mqtt_port = mqtt_port
        self.opcua_endpoint = f"opc.tcp://127.0.0.1:{opcua_port}/tvdi/lab/"
        self.namespace_index = 2
        self.broker = LabMqttBroker(self.mqtt_host, mqtt_port)
        self.opcua = OPCUAServerStub(self.opcua_endpoint)

    async def start(self) -> None:
        await self.broker.start()
        self.namespace_index = await self.opcua.start(LAB_VALUES)
        logger.info("lab line up: mqtt %s opc %s", self.mqtt_port, self.opcua_endpoint)

    async def stop(self) -> None:
        try:
            await asyncio.wait_for(self.opcua.stop(), timeout=2)
        except Exception:
            logger.warning("lab OPC UA stop timed out")
        await self.broker.stop()
