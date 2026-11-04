"""Repository برای دسترسی به داده"""
import json
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from tvdi_industrial.config import get_settings
from tvdi_industrial.persistence.models import Base, Tire, Inspection


class Repository:
    def __init__(self, database_url: str = None):
        s = get_settings()
        url = database_url or s.database_url
        if url.startswith("sqlite"):
            path = url.split("///", 1)[-1]
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.database_url = url
        self.engine = create_async_engine(url, pool_pre_ping=True)
        self.Session = async_sessionmaker(self.engine, expire_on_commit=False)
        self.ready = False

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.ready = True

    async def save_inspection(self, meta: dict, result: dict, model_version: str):
        async with self.Session() as session:
            tire = await session.scalar(select(Tire).where(Tire.tire_id == meta["tire_id"]))
            if not tire:
                tire = Tire(**{k: meta.get(k) for k in (
                    "tire_id", "batch_id", "machine_id", "mold_id",
                    "material_lot", "build_temp", "build_pressure",
                    "cure_temp", "cure_time",
                )})
                session.add(tire)
                await session.flush()

            insp = Inspection(
                tire_id=meta["tire_id"],
                defect_type=result["defect_type"],
                defect_probability=result["defect_probability"],
                severity=result.get("severity"),
                risk_score=result.get("risk_score"),
                risk_level=result.get("risk_level"),
                grade=result.get("grade"),
                recommendation=result.get("recommendation"),
                reasons=json.dumps(result.get("reasons") or [], ensure_ascii=False),
                model_version=model_version,
                model_source=result.get("source"),
                image_path=result.get("image_path"),
            )
            session.add(insp)
            await session.commit()
            return insp.id

    async def defect_rate(self, hours: int = 24) -> dict:
        since = datetime.utcnow() - timedelta(hours=hours)
        async with self.Session() as session:
            total = await session.scalar(
                select(func.count(Inspection.id)).where(Inspection.created_at >= since)
            )
            defects = await session.scalar(
                select(func.count(Inspection.id)).where(
                    Inspection.created_at >= since,
                    Inspection.defect_type != "good",
                )
            )
        return {"total": total or 0, "defects": defects or 0,
                "rate": (defects or 0) / max(total or 1, 1)}

    async def list_inspections(self, hours: int = 48) -> list[dict]:
        since = datetime.utcnow() - timedelta(hours=hours)
        stmt = (
            select(Inspection, Tire)
            .join(Tire, Tire.tire_id == Inspection.tire_id)
            .where(Inspection.created_at >= since)
            .order_by(Inspection.created_at.desc())
        )
        async with self.Session() as session:
            rows = (await session.execute(stmt)).all()
        items = []
        for insp, tire in rows:
            items.append({
                "tire_id": tire.tire_id,
                "batch_id": tire.batch_id,
                "machine_id": tire.machine_id,
                "mold_id": tire.mold_id,
                "material_lot": tire.material_lot,
                "build_temp": tire.build_temp,
                "build_pressure": tire.build_pressure,
                "cure_temp": tire.cure_temp,
                "cure_time": tire.cure_time,
                "timestamp": insp.created_at.isoformat() if insp.created_at else None,
                "defect_type": insp.defect_type,
                "defect_probability": insp.defect_probability,
                "severity": insp.severity,
                "risk_score": insp.risk_score,
                "risk_level": insp.risk_level,
                "grade": insp.grade,
                "recommendation": insp.recommendation,
                "action": insp.recommendation,
                "model_version": insp.model_version,
                "model_source": insp.model_source,
                "image_path": insp.image_path,
            })
        return items

    async def trace(self, metadata: dict, defect_type: str, top_k: int = 3) -> list[dict]:
        """Lift from stored inspections. Returns nothing until the history is large enough."""
        async with self.Session() as session:
            rows = (await session.execute(select(Inspection, Tire).join(Tire, Tire.tire_id == Inspection.tire_id))).all()
        if len(rows) < 8:
            return []
        total = len(rows)
        baseline = sum(1 for insp, _ in rows if insp.defect_type == defect_type) / total
        baseline = max(baseline, 1e-6)
        causes = []
        for field in ("machine_id", "mold_id", "material_lot", "batch_id"):
            value = metadata.get(field)
            if value is None:
                continue
            subset = [(insp, tire) for insp, tire in rows if str(getattr(tire, field) or "") == str(value)]
            if len(subset) < 3:
                continue
            rate = sum(1 for insp, _ in subset if insp.defect_type == defect_type) / len(subset)
            lift = rate / baseline
            if lift < 1.15:
                continue
            causes.append({
                "field": field,
                "value": str(value),
                "defect_rate": float(rate),
                "lift": float(lift),
                "count": len(subset),
                "evidence": "laboratory_correlation",
                "plant_cause": False,
            })
        causes.sort(key=lambda item: item["lift"], reverse=True)
        return causes[:top_k]
