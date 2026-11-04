"""صف محلی SQLite برای قطع شبکه + sync queue"""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from tvdi_industrial.config import get_settings


class EdgeBuffer:
    def __init__(self, db_path: str = None):
        self.s = get_settings()
        self.db_path = Path(db_path or self.s.edge_buffer_db)
        self._init_db()

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    created REAL NOT NULL,
                    synced INTEGER DEFAULT 0
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_synced ON events(synced, id)")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def enqueue(self, topic: str, payload: dict) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO events (payload, topic, created) VALUES (?, ?, ?)",
                (json.dumps(payload, ensure_ascii=False), topic, time.time()),
            )
            return cur.lastrowid

    def pending(self, limit: int = 100) -> list[tuple]:
        with self._conn() as c:
            cur = c.execute(
                "SELECT id, payload, topic FROM events WHERE synced=0 ORDER BY id LIMIT ?",
                (limit,),
            )
            return cur.fetchall()

    def mark_synced(self, event_id: int):
        with self._conn() as c:
            c.execute("UPDATE events SET synced=1 WHERE id=?", (event_id,))

    def purge_synced(self, older_than_s: int = 86400):
        with self._conn() as c:
            c.execute(
                "DELETE FROM events WHERE synced=1 AND created < ?",
                (time.time() - older_than_s,),
            )

    def sync(self, sender: Callable[[str, dict], bool]):
        for event_id, payload, topic in self.pending():
            try:
                if sender(topic, json.loads(payload)):
                    self.mark_synced(event_id)
            except Exception:
                break
