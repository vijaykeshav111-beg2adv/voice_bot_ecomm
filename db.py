"""MySQL access for the appointment system: a small connection pool and automatic schema set-up.

Configure with MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE in .env.
PyMySQL is imported lazily, so the rest of the bot still runs if it is not installed.
"""
import contextlib
import logging
import os
import queue
import threading
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306") or 3306)
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "voicebot")
MYSQL_POOL_SIZE = max(1, int(os.getenv("MYSQL_POOL_SIZE", "5") or 5))
SCHEMA_FILE = BASE_DIR / "schema.sql"


def _connect(with_database=True):
    import pymysql
    from pymysql.cursors import DictCursor
    kwargs = dict(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD,
                  charset="utf8mb4", cursorclass=DictCursor, connect_timeout=5, autocommit=False)
    if with_database:
        kwargs["database"] = MYSQL_DATABASE
    return pymysql.connect(**kwargs)


def is_integrity_error(exc):
    """True for a duplicate-key / constraint error from any DB-API driver (PyMySQL, sqlite3, ...)."""
    return type(exc).__name__ == "IntegrityError"


def _schema_statements():
    lines = [ln for ln in SCHEMA_FILE.read_text(encoding="utf-8").splitlines()
             if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


_ready = False
_ready_lock = threading.Lock()


def init_schema():
    """Create the database (if the MySQL user may) and the tables. Safe to call repeatedly."""
    global _ready
    with _ready_lock:
        if _ready:
            return
        if not MYSQL_DATABASE.replace("_", "").isalnum():
            raise RuntimeError("MYSQL_DATABASE may only contain letters, digits and underscores")
        try:
            conn = _connect(with_database=True)
        except Exception as exc:
            if getattr(exc, "args", (None,))[0] != 1049:        # 1049 = unknown database
                raise
            admin = _connect(with_database=False)
            try:
                with admin.cursor() as cur:
                    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
                                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                admin.commit()
            finally:
                admin.close()
            conn = _connect(with_database=True)
        try:
            with conn.cursor() as cur:
                for stmt in _schema_statements():
                    cur.execute(stmt)
                _drop_phone_column(cur)
            conn.commit()
        finally:
            conn.close()
        _ready = True
        logging.info("MySQL ready: %s@%s:%s/%s", MYSQL_USER, MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE)


def _drop_phone_column(cur):
    """Earlier versions stored a phone number. It is no longer collected, so remove the column (and its
    index) from an existing table. Runs once; does nothing on a fresh database."""
    cur.execute("SELECT COUNT(*) AS n FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name = 'appointments' AND column_name = 'phone'")
    if cur.fetchall()[0]["n"]:
        cur.execute("SELECT COUNT(*) AS n FROM information_schema.statistics "
                    "WHERE table_schema = DATABASE() AND table_name = 'appointments' AND index_name = 'idx_phone_start'")
        if cur.fetchall()[0]["n"]:
            cur.execute("ALTER TABLE appointments DROP INDEX idx_phone_start")
        cur.execute("ALTER TABLE appointments DROP COLUMN phone")
        logging.warning("Removed the old 'phone' column from the appointments table")


class _Pool:
    def __init__(self, size):
        self._idle = queue.LifoQueue()
        self._size = size
        self._made = 0
        self._lock = threading.Lock()

    def get(self):
        try:
            conn = self._idle.get_nowait()
        except queue.Empty:
            with self._lock:
                create = self._made < self._size
                if create:
                    self._made += 1
            if create:
                try:
                    return _connect()
                except Exception:
                    with self._lock:
                        self._made -= 1
                    raise
            conn = self._idle.get(timeout=10)      # every connection is busy: wait for one
        try:
            conn.ping(reconnect=True)               # the server may have closed an idle connection
        except Exception:
            self.discard(conn)
            with self._lock:
                self._made += 1
            try:
                return _connect()
            except Exception:
                with self._lock:
                    self._made -= 1
                raise
        return conn

    def put(self, conn):
        self._idle.put(conn)

    def discard(self, conn):
        with self._lock:
            self._made -= 1
        with contextlib.suppress(Exception):
            conn.close()


_pool = _Pool(MYSQL_POOL_SIZE)


@contextlib.contextmanager
def connection():
    """Borrow a connection. Commits when the block ends normally, rolls back on any exception.
    Use `with conn.cursor() as cur:` inside; rows come back as dicts."""
    init_schema()
    conn = _pool.get()
    try:
        yield conn
        conn.commit()
    except BaseException:
        try:
            conn.rollback()
        except Exception:
            _pool.discard(conn)
            raise
        _pool.put(conn)
        raise
    else:
        _pool.put(conn)


def status():
    """(ok, message) for the UI / health check. Never raises."""
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                cur.fetchall()
        return True, f"{MYSQL_USER}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    except ModuleNotFoundError:
        return False, "PyMySQL is not installed (pip install -r requirements.txt)"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
