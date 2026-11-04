"""مدل‌های پایدار برای PostgreSQL + TimescaleDB"""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, Index
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

    inspections = relationship("Inspection", back_populates="tire")


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
    reasons = Column(Text, default="")
    model_version = Column(String(32))
    model_source = Column(String(32))
    image_path = Column(String(256))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    tire = relationship("Tire", back_populates="inspections")

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
