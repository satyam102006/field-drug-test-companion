import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def pg_server(tmp_path_factory):
    pgserver = pytest.importorskip("pgserver")
    srv = pgserver.get_server(str(tmp_path_factory.mktemp("pg")), cleanup_mode="stop")
    yield srv
    srv.cleanup()


@pytest.fixture(params=["sqlite", "postgres"])
def database_url(request, tmp_path):
    """None -> SQLite in tmp_path; otherwise a fresh Postgres database URL."""
    if request.param == "sqlite":
        return None
    srv = request.getfixturevalue("pg_server")
    name = "t_" + uuid.uuid4().hex[:12]
    import psycopg

    with psycopg.connect(srv.get_uri(), autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {name}")
    return srv.get_uri(name)
