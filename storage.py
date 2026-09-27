"""
Evidence storage backends.

* SQLiteStorage   - local / offline use (single file, WAL, IMMEDIATE transactions, lock retry).
* PostgresStorage - hosted use (Supabase / any Postgres), pooled connections, advisory lock
                    serialises hash-chain appends across all app instances.

Evidence images are stored as BLOB/BYTEA in the same row as the record so that a record and its
evidence are always committed (or rolled back) atomically.
"""

from __future__ import annotations

import random
import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd

GENESIS_HASH = "0" * 64
LEDGER_COLUMNS = "id, timestamp, operator_id, gps_location, result, delta_e, hash_signature"
RECORD_FIELDS = (
    "timestamp", "operator_id", "gps_location", "result", "delta_e", "hash_signature", "hmac_signature",
    "prev_hash", "raw_image_sha256", "lab_values", "calibration_rmse", "evidence_png",
)
USER_FIELDS = ("username", "full_name", "role", "password_hash", "active", "failed_attempts", "locked_until", "created_at")

DB_MAX_RETRIES = 6
DB_TIMEOUT_S = 15.0
CHAIN_LOCK_KEY = 26231


class StorageBusyError(RuntimeError):
    """Raised when the database stays locked/unreachable after all retries."""


class Storage(ABC):
    backend_name = "abstract"

    @abstractmethod
    def append_record(self, build: Callable[[str], Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """Atomically read the chain head, build the row from it and insert. Returns (id, row)."""

    @abstractmethod
    def fetch_logs(self, operator_id: Optional[str] = None) -> pd.DataFrame: ...

    @abstractmethod
    def find_by_raw_hash(self, raw_sha256: str) -> Optional[int]: ...

    @abstractmethod
    def get_record(self, record_id: int) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def prev_hash_of(self, record_id: int) -> str: ...

    @abstractmethod
    def record_ids(self) -> List[int]: ...

    @abstractmethod
    def iter_records(self) -> Iterator[Dict[str, Any]]: ...

    @abstractmethod
    def count_users(self) -> int: ...

    @abstractmethod
    def get_user(self, username: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def list_users(self) -> pd.DataFrame: ...

    @abstractmethod
    def create_user(self, user: Dict[str, Any]) -> None: ...

    @abstractmethod
    def update_user(self, username: str, **fields: Any) -> None: ...


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
class SQLiteStorage(Storage):
    backend_name = "SQLite (local file)"

    def __init__(self, db_path: str, legacy_evidence_dir: Optional[str] = None) -> None:
        self.db_path = str(Path(db_path).resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema(legacy_evidence_dir)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_S)
        conn.execute(f"PRAGMA busy_timeout = {int(DB_TIMEOUT_S * 1000)}")
        return conn

    def _init_schema(self, legacy_evidence_dir: Optional[str]) -> None:
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            with conn:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(test_logs)")]
                if cols and "evidence_png" not in cols:
                    self._migrate_legacy(conn, legacy_evidence_dir)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS test_logs (
                        id               INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp        TEXT NOT NULL,
                        operator_id      TEXT NOT NULL,
                        gps_location     TEXT NOT NULL,
                        result           TEXT NOT NULL,
                        delta_e          REAL NOT NULL,
                        hash_signature   TEXT NOT NULL UNIQUE,
                        hmac_signature   TEXT NOT NULL,
                        prev_hash        TEXT NOT NULL,
                        raw_image_sha256 TEXT NOT NULL,
                        lab_values       TEXT NOT NULL,
                        calibration_rmse REAL NOT NULL,
                        evidence_png     BLOB NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_raw ON test_logs(raw_image_sha256)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_operator ON test_logs(operator_id)")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        username        TEXT PRIMARY KEY,
                        full_name       TEXT NOT NULL,
                        role            TEXT NOT NULL CHECK (role IN ('officer', 'admin')),
                        password_hash   TEXT NOT NULL,
                        active          INTEGER NOT NULL DEFAULT 1,
                        failed_attempts INTEGER NOT NULL DEFAULT 0,
                        locked_until    TEXT,
                        created_at      TEXT NOT NULL
                    )
                    """
                )

    @staticmethod
    def _migrate_legacy(conn: sqlite3.Connection, legacy_evidence_dir: Optional[str]) -> None:
        """Upgrade the file-based evidence schema (evidence_file column) to in-row BLOBs."""
        evidence_dir = Path(legacy_evidence_dir) if legacy_evidence_dir else None
        conn.execute("ALTER TABLE test_logs RENAME TO test_logs_legacy")
        conn.execute(
            """
            CREATE TABLE test_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, operator_id TEXT NOT NULL,
                gps_location TEXT NOT NULL, result TEXT NOT NULL, delta_e REAL NOT NULL,
                hash_signature TEXT NOT NULL UNIQUE, hmac_signature TEXT NOT NULL, prev_hash TEXT NOT NULL,
                raw_image_sha256 TEXT NOT NULL, lab_values TEXT NOT NULL, calibration_rmse REAL NOT NULL,
                evidence_png BLOB NOT NULL
            )
            """
        )
        rows = conn.execute("SELECT * FROM test_logs_legacy ORDER BY id").fetchall()
        names = [d[0] for d in conn.execute("SELECT * FROM test_logs_legacy LIMIT 0").description]
        for row in rows:
            rec = dict(zip(names, row))
            path = evidence_dir / rec["evidence_file"] if evidence_dir else None
            blob = path.read_bytes() if path and path.exists() else b""
            conn.execute(
                f"INSERT INTO test_logs (id, {', '.join(RECORD_FIELDS)}) VALUES ({', '.join('?' * (len(RECORD_FIELDS) + 1))})",
                (rec["id"], *[rec[f] for f in RECORD_FIELDS[:-1]], blob),
            )
        conn.execute("DROP TABLE test_logs_legacy")

    def _write(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(DB_MAX_RETRIES):
            try:
                with closing(self._connect()) as conn:
                    conn.isolation_level = None
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        out = fn(conn)
                        conn.execute("COMMIT")
                        return out
                    except BaseException:
                        conn.execute("ROLLBACK")
                        raise
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "locked" not in msg and "busy" not in msg:
                    raise
                last_exc = exc
                time.sleep(min(2.0, 0.05 * (2 ** attempt)) + random.uniform(0, 0.05))
        raise StorageBusyError(f"database remained locked after {DB_MAX_RETRIES} attempts: {last_exc}")

    def _read(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql, params).fetchall()

    def append_record(self, build):
        def _tx(conn: sqlite3.Connection):
            head = conn.execute("SELECT hash_signature FROM test_logs ORDER BY id DESC LIMIT 1").fetchone()
            row = build(head[0] if head else GENESIS_HASH)
            cur = conn.execute(
                f"INSERT INTO test_logs ({', '.join(RECORD_FIELDS)}) VALUES ({', '.join('?' * len(RECORD_FIELDS))})",
                tuple(row[f] for f in RECORD_FIELDS),
            )
            return int(cur.lastrowid), row

        return self._write(_tx)

    def fetch_logs(self, operator_id=None):
        with closing(self._connect()) as conn:
            if operator_id is None:
                return pd.read_sql_query(f"SELECT {LEDGER_COLUMNS} FROM test_logs ORDER BY id DESC", conn)
            return pd.read_sql_query(
                f"SELECT {LEDGER_COLUMNS} FROM test_logs WHERE operator_id = ? ORDER BY id DESC", conn, params=(operator_id,)
            )

    def find_by_raw_hash(self, raw_sha256):
        rows = self._read("SELECT id FROM test_logs WHERE raw_image_sha256 = ? LIMIT 1", (raw_sha256,))
        return int(rows[0][0]) if rows else None

    def get_record(self, record_id):
        rows = self._read("SELECT * FROM test_logs WHERE id = ?", (int(record_id),))
        return dict(rows[0]) if rows else None

    def prev_hash_of(self, record_id):
        rows = self._read("SELECT hash_signature FROM test_logs WHERE id < ? ORDER BY id DESC LIMIT 1", (int(record_id),))
        return rows[0][0] if rows else GENESIS_HASH

    def record_ids(self):
        return [int(r[0]) for r in self._read("SELECT id FROM test_logs ORDER BY id")]

    def iter_records(self):
        for rid in self.record_ids():
            rec = self.get_record(rid)
            if rec:
                yield rec

    def count_users(self):
        return int(self._read("SELECT COUNT(*) FROM users")[0][0])

    def get_user(self, username):
        rows = self._read("SELECT * FROM users WHERE username = ?", (username,))
        if not rows:
            return None
        user = dict(rows[0])
        user["active"] = bool(user["active"])
        return user

    def list_users(self):
        with closing(self._connect()) as conn:
            df = pd.read_sql_query(
                "SELECT username, full_name, role, active, failed_attempts, locked_until, created_at FROM users ORDER BY username",
                conn,
            )
        df["active"] = df["active"].astype(bool)
        return df

    def create_user(self, user):
        vals = {**user, "active": int(bool(user.get("active", True)))}
        self._write(lambda c: c.execute(
            f"INSERT INTO users ({', '.join(USER_FIELDS)}) VALUES ({', '.join('?' * len(USER_FIELDS))})",
            tuple(vals.get(f) for f in USER_FIELDS),
        ))

    def update_user(self, username, **fields):
        if not fields:
            return
        if "active" in fields:
            fields["active"] = int(bool(fields["active"]))
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._write(lambda c: c.execute(f"UPDATE users SET {sets} WHERE username = ?", (*fields.values(), username)))


# ---------------------------------------------------------------------------
# PostgreSQL (Supabase)
# ---------------------------------------------------------------------------
class PostgresStorage(Storage):
    backend_name = "PostgreSQL (Supabase)"

    def __init__(self, database_url: str) -> None:
        import psycopg
        from psycopg_pool import ConnectionPool

        self._psycopg = psycopg
        self.pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=5,
            kwargs={"prepare_threshold": None, "connect_timeout": 15},
            check=ConnectionPool.check_connection,
            open=True,
            timeout=20,
        )
        self._init_schema()

    def _run(self, fn: Callable[[Any], Any]) -> Any:
        psycopg = self._psycopg
        last_exc: Optional[Exception] = None
        for attempt in range(DB_MAX_RETRIES):
            try:
                with self.pool.connection() as conn:
                    with conn.transaction():
                        return fn(conn)
            except (psycopg.OperationalError, psycopg.errors.SerializationFailure,
                    psycopg.errors.DeadlockDetected) as exc:
                last_exc = exc
                time.sleep(min(2.0, 0.1 * (2 ** attempt)) + random.uniform(0, 0.1))
        raise StorageBusyError(f"database unavailable after {DB_MAX_RETRIES} attempts: {last_exc}")

    def _rows(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        from psycopg.rows import dict_row

        def _q(conn):
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return cur.fetchall()

        return self._run(_q)

    def _init_schema(self) -> None:
        def _tx(conn):
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (CHAIN_LOCK_KEY + 1,))
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_logs (
                    id               BIGSERIAL PRIMARY KEY,
                    timestamp        TEXT NOT NULL,
                    operator_id      TEXT NOT NULL,
                    gps_location     TEXT NOT NULL,
                    result           TEXT NOT NULL,
                    delta_e          DOUBLE PRECISION NOT NULL,
                    hash_signature   TEXT NOT NULL UNIQUE,
                    hmac_signature   TEXT NOT NULL,
                    prev_hash        TEXT NOT NULL,
                    raw_image_sha256 TEXT NOT NULL,
                    lab_values       TEXT NOT NULL,
                    calibration_rmse DOUBLE PRECISION NOT NULL,
                    evidence_png     BYTEA NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_raw ON test_logs(raw_image_sha256)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_operator ON test_logs(operator_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username        TEXT PRIMARY KEY,
                    full_name       TEXT NOT NULL,
                    role            TEXT NOT NULL CHECK (role IN ('officer', 'admin')),
                    password_hash   TEXT NOT NULL,
                    active          BOOLEAN NOT NULL DEFAULT TRUE,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until    TEXT,
                    created_at      TEXT NOT NULL
                )
                """
            )
            # Supabase exposes the public schema over its REST API; RLS with no policies blocks that path.
            conn.execute("ALTER TABLE test_logs ENABLE ROW LEVEL SECURITY")
            conn.execute("ALTER TABLE users ENABLE ROW LEVEL SECURITY")

        self._run(_tx)

    def append_record(self, build):
        def _tx(conn):
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (CHAIN_LOCK_KEY,))
            head = conn.execute("SELECT hash_signature FROM test_logs ORDER BY id DESC LIMIT 1").fetchone()
            row = build(head[0] if head else GENESIS_HASH)
            new_id = conn.execute(
                f"INSERT INTO test_logs ({', '.join(RECORD_FIELDS)}) VALUES ({', '.join(['%s'] * len(RECORD_FIELDS))}) RETURNING id",
                tuple(row[f] for f in RECORD_FIELDS),
            ).fetchone()[0]
            return int(new_id), row

        return self._run(_tx)

    def fetch_logs(self, operator_id=None):
        if operator_id is None:
            rows = self._rows(f"SELECT {LEDGER_COLUMNS} FROM test_logs ORDER BY id DESC")
        else:
            rows = self._rows(f"SELECT {LEDGER_COLUMNS} FROM test_logs WHERE operator_id = %s ORDER BY id DESC", (operator_id,))
        return pd.DataFrame(rows, columns=[c.strip() for c in LEDGER_COLUMNS.split(",")])

    def find_by_raw_hash(self, raw_sha256):
        rows = self._rows("SELECT id FROM test_logs WHERE raw_image_sha256 = %s LIMIT 1", (raw_sha256,))
        return int(rows[0]["id"]) if rows else None

    def get_record(self, record_id):
        rows = self._rows("SELECT * FROM test_logs WHERE id = %s", (int(record_id),))
        if not rows:
            return None
        rec = dict(rows[0])
        rec["evidence_png"] = bytes(rec["evidence_png"])
        return rec

    def prev_hash_of(self, record_id):
        rows = self._rows("SELECT hash_signature FROM test_logs WHERE id < %s ORDER BY id DESC LIMIT 1", (int(record_id),))
        return rows[0]["hash_signature"] if rows else GENESIS_HASH

    def record_ids(self):
        return [int(r["id"]) for r in self._rows("SELECT id FROM test_logs ORDER BY id")]

    def iter_records(self):
        for rid in self.record_ids():
            rec = self.get_record(rid)
            if rec:
                yield rec

    def count_users(self):
        return int(self._rows("SELECT COUNT(*) AS n FROM users")[0]["n"])

    def get_user(self, username):
        rows = self._rows("SELECT * FROM users WHERE username = %s", (username,))
        return dict(rows[0]) if rows else None

    def list_users(self):
        rows = self._rows(
            "SELECT username, full_name, role, active, failed_attempts, locked_until, created_at FROM users ORDER BY username"
        )
        return pd.DataFrame(rows, columns=["username", "full_name", "role", "active", "failed_attempts", "locked_until", "created_at"])

    def create_user(self, user):
        self._run(lambda c: c.execute(
            f"INSERT INTO users ({', '.join(USER_FIELDS)}) VALUES ({', '.join(['%s'] * len(USER_FIELDS))})",
            tuple(bool(user.get(f, True)) if f == "active" else user.get(f) for f in USER_FIELDS),
        ))

    def update_user(self, username, **fields):
        if not fields:
            return
        sets = ", ".join(f"{k} = %s" for k in fields)
        self._run(lambda c: c.execute(f"UPDATE users SET {sets} WHERE username = %s", (*fields.values(), username)))


def open_storage(database_url: Optional[str], sqlite_path: str, legacy_evidence_dir: Optional[str] = None) -> Storage:
    if database_url:
        return PostgresStorage(database_url)
    return SQLiteStorage(sqlite_path, legacy_evidence_dir)
