"""ارسال دستور Reject با تأییدیه دوگانه (TVDI + PLC) و watchdog"""
import asyncio
import logging
from datetime import datetime
from typing import Optional

from tvdi_industrial.config import get_settings
from tvdi_industrial.industrial.opcua_client import OPCUAClient

logger = logging.getLogger(__name__)


class RejectActuator:
    def __init__(self, opcua: OPCUAClient, settings=None):
        self.s = settings or get_settings()
        self.opcua = opcua
        self._last_reject: Optional[datetime] = None

    async def reject(self, tire_id: str, reason: str) -> dict:
        if not self.s.reject_enabled:
            logger.info("reject disabled (dry-run) for %s: %s", tire_id, reason)
            return {"tire_id": tire_id, "status": "dry_run", "reason": reason}

        tvdi_ok = await self.opcua.send_reject(tire_id)
        if not tvdi_ok:
            logger.error("TVDI reject command failed for %s", tire_id)
            return {"tire_id": tire_id, "status": "failed", "stage": "tvdi_send"}

        plc_ok = await self._await_plc_confirmation(timeout_ms=self.s.reject_plc_confirm_timeout_ms)
        self._last_reject = datetime.utcnow()

        if not plc_ok:
            logger.error("PLC did not confirm reject for %s", tire_id)
            return {"tire_id": tire_id, "status": "failed", "stage": "plc_confirm"}

        logger.info("Reject confirmed: %s (%s)", tire_id, reason)
        return {"tire_id": tire_id, "status": "confirmed", "reason": reason}

    async def _await_plc_confirmation(self, timeout_ms: int) -> bool:
        deadline = asyncio.get_event_loop().time() + timeout_ms / 1000
        while asyncio.get_event_loop().time() < deadline:
            try:
                ack = await self.opcua.read_node("ns=2;s=Reject.Ack")
                if ack:
                    await self.opcua.write_node("ns=2;s=Reject.Trigger", False)
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.05)
        return False
