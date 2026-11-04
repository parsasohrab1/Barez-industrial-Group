import asyncio

import numpy as np
import pytest

from tvdi_industrial.cycle import software_cycle
from tvdi_industrial.decision import LAB_LIMITS, factory_grade, station_decision
from tvdi_industrial.industrial.lab_broker import LabMqttBroker
from tvdi_industrial.localize import box_iou


def test_factory_grade_stays_incomplete_without_geometry():
    grade = factory_grade({}, None)
    assert grade["status"] == "incomplete"
    station = station_decision("A", grade)
    assert station["status"] == "incomplete"
    assert station["grade"] == "A"


def test_factory_grade_uses_only_the_lab_limit_sheet():
    measurements = {key: 1.0 for key in (
        "rfv_n", "lfv_n", "static_imbalance_g", "couple_imbalance_gcm",
        "bulge_mm", "depression_mm", "radial_runout_mm",
    )}
    measurements["rfv_n"] = 120
    grade = factory_grade(measurements, LAB_LIMITS)
    assert grade["status"] == "complete"
    assert grade["grade"] == "C"
    assert grade["limit_source"] == "lab_fixture_not_oem"
    assert station_decision("A", grade)["grade"] == "C"


def test_software_cycle_is_not_a_machine_cycle():
    calls = {"n": 0}

    def score(_canvas):
        calls["n"] += 1
        return "ok"

    report = software_cycle(np.zeros((128, 128), dtype=np.uint8), score)
    assert calls["n"] == 16
    assert report["motion_measured"] is False
    assert report["software_cycle_ms"] >= 0
    assert report["published_reference"]["xray_machine_cycle_claim_s"] == 22


def test_box_iou_is_one_for_the_same_box():
    box = (0.5, 0.5, 0.2, 0.4)
    assert box_iou(box, box) == pytest.approx(1.0)
    assert box_iou(box, (0.1, 0.1, 0.05, 0.05)) == 0.0


def test_lab_broker_accepts_a_publish():
    async def scenario():
        broker = LabMqttBroker("127.0.0.1", 0)
        await broker.start()
        port = broker._server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(bytes([0x10, 0x02, 0x00, 0x00]))
            topic = b"lab/tire"
            body = struct_publish(topic, b"{}")
            writer.write(body)
            await writer.drain()
            ack = await asyncio.wait_for(reader.read(64), timeout=2)
            writer.close()
            await writer.wait_closed()
            return ack, broker.messages
        finally:
            await broker.stop()

    ack, messages = asyncio.run(scenario())
    assert messages and messages[0][0] == "lab/tire"
    assert ack[0] == 0x20 or 0x40 in ack


def struct_publish(topic: bytes, payload: bytes) -> bytes:
    packet = len(topic).to_bytes(2, "big") + topic + b"\x00\x01" + payload
    return bytes([0x32]) + bytes([len(packet)]) + packet
