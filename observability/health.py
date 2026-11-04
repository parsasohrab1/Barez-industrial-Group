"""Deep health check با بررسی همه وابستگی‌ها"""
import asyncio
from typing import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


class HealthRegistry:
    def __init__(self):
        self._checks: dict[str, Callable] = {}

    def register(self, name: str, check: Callable):
        self._checks[name] = check

    async def run(self) -> dict:
        results = {}
        for name, check in self._checks.items():
            try:
                if asyncio.iscoroutinefunction(check):
                    await check()
                else:
                    check()
                results[name] = "ok"
            except Exception as exc:
                results[name] = f"fail: {exc}"
        overall = "ok" if all(v == "ok" for v in results.values()) else "degraded"
        return {"status": overall, "checks": results}


health = HealthRegistry()


@router.get("/health/live")
async def live():
    return {"status": "ok"}


@router.get("/health/ready")
async def ready():
    return await health.run()


@router.get("/health/deep")
async def deep():
    data = await health.run()
    code = 200 if data["status"] == "ok" else 503
    return JSONResponse(content=data, status_code=code)
