"""
Copy local SQLite records + accounts into Postgres (Supabase), preserving record ids so the hash chain
stays valid, then verify every record in the target.

Usage:
    python scripts/migrate_to_postgres.py "postgresql://...pooler.supabase.com:5432/postgres?sslmode=require"

Records remain verifiable only if the hosted app uses the same device key; the script prints the local
key so it can be pasted into the DEVICE_HMAC_KEY secret.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import FieldTestPipeline, load_or_create_key  # noqa: E402
from storage import RECORD_FIELDS, USER_FIELDS, PostgresStorage, SQLiteStorage  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    data = ROOT / "data"
    src = SQLiteStorage(str(data / "field_tests.db"), legacy_evidence_dir=str(data / "evidence"))
    dst = PostgresStorage(sys.argv[1])
    if dst.record_ids():
        print("Target already contains records - refusing to merge chains. Use an empty database.")
        return 1

    records = list(src.iter_records())
    users = src.list_users()["username"].tolist()

    def _copy(conn):
        for rec in records:
            conn.execute(
                f"INSERT INTO test_logs (id, {', '.join(RECORD_FIELDS)}) VALUES ({', '.join(['%s'] * (len(RECORD_FIELDS) + 1))})",
                (rec["id"], *[rec[f] for f in RECORD_FIELDS]),
            )
        conn.execute("SELECT setval(pg_get_serial_sequence('test_logs', 'id'), COALESCE(MAX(id), 1), MAX(id) IS NOT NULL) FROM test_logs")
        for name in users:
            u = src.get_user(name)
            conn.execute(
                f"INSERT INTO users ({', '.join(USER_FIELDS)}) VALUES ({', '.join(['%s'] * len(USER_FIELDS))}) "
                "ON CONFLICT (username) DO NOTHING",
                tuple(u[f] for f in USER_FIELDS),
            )

    dst._run(_copy)
    key = load_or_create_key(data / ".device_key")
    total, failed = FieldTestPipeline(storage=dst, device_key=key).verify_chain()
    print(f"Copied {len(records)} records and {len(users)} accounts. Verified {total}, failed: {failed or 'none'}")
    print(f"Set this Streamlit secret to keep signatures valid:\nDEVICE_HMAC_KEY = \"{key.hex()}\"")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
