"""Test scaffolding.

MySQL is replaced by SQLite behind the same `db.connection()` interface, so the booking logic runs for
real (including the unique key that stops double-booking). Only the table DDL differs from schema.sql.
The LLM and the text-to-speech service are faked where a test needs them.
"""
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import booking  # noqa: E402
import core  # noqa: E402
import db  # noqa: E402

DDL = """
CREATE TABLE appointments (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ref_code TEXT NOT NULL UNIQUE, customer_name TEXT NOT NULL,
  purpose TEXT NULL, slot_start TEXT NOT NULL, slot_end TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'booked', active INTEGER NULL DEFAULT 1, channel TEXT NOT NULL DEFAULT 'web',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE (slot_start, active))
"""
_DT_COLUMNS = {"slot_start", "slot_end", "created_at", "updated_at"}
FMT = "%Y-%m-%d %H:%M:%S"


class ShimCursor:
    """Looks like a PyMySQL DictCursor: %s placeholders, datetime in/out, dict rows."""

    def __init__(self, conn):
        self._c = conn.cursor()

    def execute(self, sql, params=()):
        params = [p.strftime(FMT) if isinstance(p, datetime) else p for p in (params or ())]
        self._c.execute(sql.replace("%s", "?"), params)

    def fetchall(self):
        cols = [d[0] for d in (self._c.description or [])]
        out = []
        for row in self._c.fetchall():
            d = dict(zip(cols, row))
            for k in _DT_COLUMNS & d.keys():
                d[k] = datetime.strptime(d[k], FMT)
            out.append(d)
        return out

    @property
    def lastrowid(self):
        return self._c.lastrowid

    @property
    def rowcount(self):
        return self._c.rowcount

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._c.close()


class ShimConn:
    def __init__(self, raw):
        self.raw = raw

    def cursor(self):
        return ShimCursor(self.raw)


@pytest.fixture
def clock(monkeypatch):
    """Frozen 'now': Friday 23 Oct 2026, 10:00 at the business. Mutable via clock['now']."""
    state = {"now": datetime(2026, 10, 23, 10, 0)}
    monkeypatch.setattr(booking, "now", lambda: state["now"])
    return state


@pytest.fixture
def sqlite_db(tmp_path, monkeypatch, clock):
    path = tmp_path / "test.db"
    setup = sqlite3.connect(path)
    setup.execute(DDL)
    setup.commit()
    setup.close()

    @contextmanager
    def connection():
        raw = sqlite3.connect(path, timeout=15)
        try:
            yield ShimConn(raw)
            raw.commit()
        except BaseException:
            raw.rollback()
            raise
        finally:
            raw.close()

    monkeypatch.setattr(db, "connection", connection)
    return path


@pytest.fixture(autouse=True)
def isolate_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "METADATA_JSONL", tmp_path / "meta.jsonl")
    monkeypatch.setattr(core, "METADATA_CSV", tmp_path / "meta.csv")
