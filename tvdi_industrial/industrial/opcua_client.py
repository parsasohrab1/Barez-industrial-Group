"""اتصال به PLC خط ساخت/پخت از طریق OPC UA
- خواندن دما/فشار/زمان پخت
- نوشتن دستور Reject
- reconnect خودکار
"""
import asyncio
import logging
from typing import Any, Callable, Optional

from asyncua import Client, Server, ua

from tvdi_industrial.config import get_settings

logger = logging.getLogger(__name__)


class OPCUAClient:
    """کلاینت async برای PLC صنعتی"""

    def __init__(self, settings=None):
        self.s = settings or get_settings()
        self._client: Optional[Client] = None
        self._lock = asyncio.Lock()
        self._reconnect_task: Optional[asyncio.Task] = None
        self._nodes: dict[str, Any] = {}

    async def connect(self) -> None:
        async with self._lock:
            if self._client and await self._is_connected():
                return
            client = Client(url=self.s.opcua_endpoint)
            if self.s.opcua_username:
                client.set_user(self.s.opcua_username)
                client.set_password(self.s.opcua_password)
            await client.connect()
            self._client = client
            logger.info("OPC UA connected: %s", self.s.opcua_endpoint)
            self._start_watchdog()

    async def _is_connected(self) -> bool:
        try:
            await self._client.get_root_node().read_browse_name()
            return True
        except Exception:
            return False

    def _start_watchdog(self):
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._watchdog())

    async def _watchdog(self):
        while True:
            await asyncio.sleep(self.s.opcua_reconnect_s)
            if not await self._is_connected():
                logger.warning("OPC UA dropped, reconnecting…")
                try:
                    await self.connect()
                except Exception as exc:
                    logger.error("OPC UA reconnect failed: %s", exc)

    async def read_node(self, node_id: str) -> Any:
        await self.connect()
        node = self._client.get_node(node_id)
        return await node.read_value()

    async def write_node(self, node_id: str, value: Any) -> None:
        await self.connect()
        node = self._client.get_node(node_id)
        await node.write_value(value)

    async def read_process_data(self) -> dict:
        """خواندن ۴ پارامتر اصلی خط ساخت/پخت"""
        line = await self.read_line_data()
        return {key: line.get(key) for key in (
            "build_temp", "build_pressure", "cure_temp", "cure_time",
        )}

    async def read_line_data(self) -> dict:
        """Process tags plus uniformity, balance, and laser geometry."""
        index = self.s.opcua_namespace_index
        mapping = {
            "build_temp": "Build.Temp",
            "build_pressure": "Build.Pressure",
            "cure_temp": "Cure.Temp",
            "cure_time": "Cure.Time",
            "rfv_n": "Uniformity.RFV",
            "lfv_n": "Uniformity.LFV",
            "static_imbalance_g": "Balance.Static",
            "couple_imbalance_gcm": "Balance.Couple",
            "bulge_mm": "Laser.Bulge",
            "depression_mm": "Laser.Depression",
            "radial_runout_mm": "Laser.RadialRunout",
        }
        result = {}
        for key, name in mapping.items():
            try:
                result[key] = float(await self.read_node(f"ns={index};s={name}"))
            except Exception as exc:
                logger.warning("read %s failed: %s", key, exc)
                result[key] = None
        return result

    async def send_reject(self, tire_id: str) -> bool:
        """ارسال دستور Reject به PLC"""
        try:
            index = self.s.opcua_namespace_index
            await self.write_node(f"ns={index};s=Reject.TireId", tire_id)
            await self.write_node(f"ns={index};s=Reject.Trigger", True)
            logger.info("Reject sent: %s", tire_id)
            return True
        except Exception as exc:
            logger.error("Reject failed: %s", exc)
            return False

    async def close(self):
        if self._reconnect_task:
            self._reconnect_task.cancel()
        if self._client:
            await self._client.disconnect()


class OPCUAServerStub:
    """Laboratory OPC UA server. Node values are fixture constants."""

    def __init__(self, endpoint="opc.tcp://127.0.0.1:4840/tvdi/lab/"):
        self.endpoint = endpoint
        self._server: Optional[Server] = None
        self.namespace_index = 2

    async def start(self, values: dict | None = None) -> int:
        server = Server()
        await server.init()
        server.set_endpoint(self.endpoint)
        server.set_server_name("TVDI Lab Fixture")
        index = await server.register_namespace("urn:tvdi:lab")
        self.namespace_index = int(index)
        folder = await server.nodes.objects.add_object(index, "Line")
        for name, value in (values or {}).items():
            node_id = ua.NodeId(name, index)
            variable = await folder.add_variable(node_id, name, value)
            await variable.set_writable()
        reject_id = await folder.add_variable(ua.NodeId("Reject.TireId", index), "Reject.TireId", "")
        await reject_id.set_writable()
        await server.start()
        self._server = server
        return self.namespace_index

    async def stop(self):
        if self._server:
            await self._server.stop()
            self._server = None
