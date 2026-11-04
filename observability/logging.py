"""لاگ ساختاریافته JSON با correlation_id"""
import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

from tvdi_industrial.config import get_settings

correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def _add_corr(_, __, event_dict):
    cid = correlation_id.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def setup_logging():
    s = get_settings()
    logging.basicConfig(stream=sys.stdout, level=s.log_level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_corr,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, s.log_level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def new_correlation_id() -> str:
    cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def get_logger(name: str):
    return structlog.get_logger(name)
