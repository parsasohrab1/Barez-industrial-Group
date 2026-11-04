"""Modbus TCP برای PLCهای قدیمی"""
import logging
from typing import Optional

from tvdi_industrial.config import get_settings

logger = logging.getLogger(__name__)


class ModbusClient:
    def __init__(self, settings=None):
        self.s = settings or get_settings()
        self._client: Optional[ModbusTcpClient] = None

    def connect(self):
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError as exc:
            raise RuntimeError("install pymodbus to use Modbus TCP") from exc
        self._client = ModbusTcpClient(self.s.modbus_host, port=self.s.modbus_port)
        self._client.connect()

    def read_register(self, address: int) -> Optional[int]:
        try:
            r = self._client.read_holding_registers(address, count=1, slave=self.s.modbus_unit_id)
            return r.registers[0]
        except Exception as exc:
            logger.error("modbus read failed: %s", exc)
            return None

    def write_register(self, address: int, value: int) -> bool:
        try:
            r = self._client.write_register(address, value, slave=self.s.modbus_unit_id)
            return not r.isError()
        except Exception as exc:
            logger.error("modbus write failed: %s", exc)
            return False

    def close(self):
        if self._client:
            self._client.close()
