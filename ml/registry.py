"""Model Registry ساده با versioning و stage"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from tvdi_industrial.config import get_settings


class ModelRegistry:
    def __init__(self, base_dir: Optional[str] = None):
        s = get_settings()
        self.base = Path(base_dir or s.model_registry_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.index = self.base / "index.json"
        if not self.index.exists():
            self.index.write_text(json.dumps({"models": []}, ensure_ascii=False))

    def _load(self) -> dict:
        return json.loads(self.index.read_text())

    def _save(self, data: dict):
        self.index.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def register(self, model_file: Path, version: str, stage: str = "staging", metrics: dict = None):
        target_dir = self.base / version
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / model_file.name
        shutil.copy2(model_file, target)

        data = self._load()
        data["models"].append({
            "version": version,
            "stage": stage,
            "path": str(target),
            "metrics": metrics or {},
            "registered_at": datetime.utcnow().isoformat(),
        })
        self._save(data)

    def promote(self, version: str, stage: str = "production"):
        data = self._load()
        for m in data["models"]:
            if m["stage"] == "production":
                m["stage"] = "archived"
            if m["version"] == version:
                m["stage"] = stage
        self._save(data)

    def get_production(self) -> Optional[dict]:
        data = self._load()
        for m in data["models"]:
            if m["stage"] == "production":
                return m
        return None
