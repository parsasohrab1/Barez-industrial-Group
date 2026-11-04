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
                    value = await check()
                else:
                    value = check()
            except Exception as exc:
                results[name] = f"fail: {exc}"
                continue
            if value is False:
                results[name] = "fail"
            elif value is True or value is None or value == "ok":
                results[name] = "ok"
            else:
                results[name] = str(value)
        failed = [v for v in results.values() if v == "fail" or str(v).startswith("fail")]
        overall = "ok" if results and not failed and all(v == "ok" for v in results.values()) else "degraded"
        if not results:
            overall = "ok"
        return {"status": overall, "checks": results}


health = HealthRegistry()


@router.get("/health/live")
async def live():
    return {"status": "ok"}


@router.get("/health/ready")
async def ready():
    data = await health.run()
    checks = data.get("checks", {})
    critical = ("database", "inference")
    broken = any(checks.get(name) == "fail" or str(checks.get(name, "")).startswith("fail") for name in critical)
    code = 503 if broken or not checks else 200
    return JSONResponse(content=data, status_code=code)


@router.get("/health/deep")
async def deep():
    data = await health.run()
    code = 200 if data["status"] == "ok" else 503
    return JSONResponse(content=data, status_code=code)
