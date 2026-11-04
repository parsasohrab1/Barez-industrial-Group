"""مدل‌های پایدار برای PostgreSQL + TimescaleDB"""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Index
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Tire(Base):
    __tablename__ = "tires"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tire_id = Column(String(64), unique=True, index=True, nullable=False)
    batch_id = Column(String(32), index=True)
    machine_id = Column(String(32), index=True)
    mold_id = Column(String(32))
    material_lot = Column(String(32))
    build_temp = Column(Float)
    build_pressure = Column(Float)
    cure_temp = Column(Float)
    cure_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    inspection = relationship("Inspection", back_populates="tire", uselist=False)


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tire_id = Column(String(64), ForeignKey("tires.tire_id"), nullable=False)
    defect_type = Column(String(32), index=True)
    defect_probability = Column(Float)
    severity = Column(String(16))
    risk_score = Column(Float)
    risk_level = Column(String(16))
    grade = Column(String(4))
    recommendation = Column(String(128))
    model_version = Column(String(32))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    tire = relationship("Tire", back_populates="inspection")

    __table_args__ = (
        Index("ix_insp_defect_created", "defect_type", "created_at"),
    )


class AlertEvent(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tire_id = Column(String(64), index=True)
    severity = Column(String(16))
    message = Column(String(512))
    acknowledged = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
