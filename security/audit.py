"""Audit Log تغییرناپذیر با hash chain برای انطباق بازرسی"""
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


class AuditLog:
    def __init__(self, db_path: str = "audit.sqlite"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    details TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    hash TEXT NOT NULL
                )
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _last_hash(self) -> str:
        with self._conn() as c:
            row = c.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            return row[0] if row else "GENESIS"

    def log(self, actor: str, action: str, resource: str, details: dict) -> str:
        prev = self._last_hash()
        ts = time.time()
        body = json.dumps(details, sort_keys=True, ensure_ascii=False)
        raw = f"{ts}|{actor}|{action}|{resource}|{body}|{prev}"
        h = hashlib.sha256(raw.encode()).hexdigest()
        with self._conn() as c:
            c.execute(
                "INSERT INTO audit (ts, actor, action, resource, details, prev_hash, hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, actor, action, resource, body, prev, h),
            )
        return h

    def verify_chain(self) -> bool:
        prev = "GENESIS"
        with self._conn() as c:
            for row in c.execute("SELECT ts, actor, action, resource, details, hash FROM audit ORDER BY id"):
                ts, actor, action, resource, body, stored = row
                raw = f"{ts}|{actor}|{action}|{resource}|{body}|{prev}"
                if hashlib.sha256(raw.encode()).hexdigest() != stored:
                    return False
                prev = stored
        return True
