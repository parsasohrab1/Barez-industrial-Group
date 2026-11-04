"""Repository برای دسترسی به داده"""
from datetime import datetime, timedelta

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from tvdi_industrial.config import get_settings
from tvdi_industrial.persistence.models import Base, Tire, Inspection, AlertEvent


class Repository:
    def __init__(self, database_url: str = None):
        s = get_settings()
        self.engine = create_async_engine(database_url or s.database_url, pool_pre_ping=True)
        self.Session = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

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
                model_version=model_version,
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
