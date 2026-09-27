import threading

import numpy as np
import pytest

from card import render_reference_card, simulate_capture
from pipeline import CHEMICAL_ANCHORS_BGR, MARKER_ERROR, FieldTestPipeline
from storage import PostgresStorage, SQLiteStorage


def _raw_sql(storage, sql, params=()):
    if isinstance(storage, SQLiteStorage):
        import sqlite3

        with sqlite3.connect(storage.db_path) as conn:
            conn.execute(sql, params)
    else:
        with storage.pool.connection() as conn:
            conn.execute(sql.replace("?", "%s"), params)


@pytest.fixture()
def pipe(tmp_path, database_url):
    storage = PostgresStorage(database_url) if database_url else SQLiteStorage(str(tmp_path / "t.db"))
    return FieldTestPipeline(storage=storage, device_key=b"k" * 32)


def capture(test_bgr, gains=(1.0, 1.0, 1.0), **kw):
    return simulate_capture(render_reference_card(scale=3, test_bgr=test_bgr), channel_gains=gains, **kw)


@pytest.mark.parametrize("name", list(CHEMICAL_ANCHORS_BGR))
@pytest.mark.parametrize("gains", [(1.0, 1.0, 1.0), (0.55, 0.8, 1.15), (1.1, 0.9, 0.6)])
def test_classifies_under_colour_cast(pipe, name, gains):
    res, err = pipe.process_image(capture(CHEMICAL_ANCHORS_BGR[name], gains), "OFF-1", "28.6139, 77.2090")
    assert err is None, err
    assert res["verdict"] == name
    assert res["delta_e"] < 35.0


def test_inconclusive(pipe):
    res, err = pipe.process_image(capture((128, 128, 128)), "OFF-1", "0,0")
    assert err is None and res["verdict"] == "Inconclusive"


def test_occluded_marker(pipe):
    res, err = pipe.process_image(capture((0, 69, 255), occlude_corner=True), "OFF-1", "0,0")
    assert res is None and err.startswith(MARKER_ERROR)


def test_glare(pipe):
    res, err = pipe.process_image(capture((0, 69, 255), glare=True), "OFF-1", "0,0")
    assert res is None
    assert "glare" in err.lower() or "uniform" in err.lower()


def test_corrupt_and_blank(pipe):
    assert pipe.process_image(None, "A", "0,0")[1].startswith("Error")
    assert pipe.process_image(np.zeros((500, 500, 3), np.uint8), "A", "0,0")[1].startswith("Error")
    noise = np.random.default_rng(0).integers(0, 255, (600, 600, 3), dtype=np.uint8)
    assert pipe.process_image(noise, "A", "0,0")[1].startswith(MARKER_ERROR)


def test_seal_verification_and_tamper(pipe):
    for i in range(3):
        _, err = pipe.process_image(capture((112, 25, 25), seed=i), f"OFF-{i}", "0,0")
        assert err is None
    total, failed = pipe.verify_chain()
    assert total == 3 and failed == []

    _raw_sql(pipe.storage, "UPDATE test_logs SET result = 'Negative_Blank' WHERE id = 2")
    assert not pipe.verify_record(2)["ok"]

    data = bytearray(pipe.get_record(3)["evidence_png"])
    data[-20] ^= 0xFF
    _raw_sql(pipe.storage, "UPDATE test_logs SET evidence_png = ? WHERE id = 3", (bytes(data),))
    assert not pipe.verify_record(3)["ok"]

    _raw_sql(pipe.storage, "DELETE FROM test_logs WHERE id = 1")
    assert not pipe.verify_record(2)["checks"]["Hash-chain link to previous record"]


def test_wrong_device_key_fails_signature(pipe):
    _, err = pipe.process_image(capture((0, 69, 255)), "OFF-1", "0,0")
    assert err is None
    other = FieldTestPipeline(storage=pipe.storage, device_key=b"x" * 32)
    report = other.verify_record(1)
    assert report["checks"]["SHA-256 seal matches image + metadata"]
    assert not report["checks"]["HMAC device signature valid"]


def test_concurrent_writes(pipe):
    img = capture((224, 255, 255))
    errors = []

    def worker(k):
        _, err = pipe.process_image(img, f"OFF-{k}", "0,0")
        if err:
            errors.append(err)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    total, failed = pipe.verify_chain()
    assert total == 8 and failed == []
