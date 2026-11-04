import pytest
from tvdi_industrial.persistence.repository import Repository


@pytest.mark.asyncio
async def test_save_and_stats(tmp_path):
    repo = Repository(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await repo.init_db()
    meta = {"tire_id": "T1", "batch_id": "B1", "machine_id": "M1",
            "mold_id": "D1", "material_lot": "L1",
            "build_temp": 155, "build_pressure": 11, "cure_temp": 165, "cure_time": 15}
    result = {"defect_type": "good", "defect_probability": 0.05,
              "severity": "low", "risk_score": 0.05, "risk_level": "low",
              "grade": "A", "recommendation": "accept"}
    await repo.save_inspection(meta, result, "v1")
    stats = await repo.defect_rate(hours=24)
    assert stats["total"] == 1
    assert stats["defects"] == 0
