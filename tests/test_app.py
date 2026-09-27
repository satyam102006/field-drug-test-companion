from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
ADMIN = ("admin", "Admin@12345")


@pytest.fixture()
def env(tmp_path, monkeypatch, database_url):
    monkeypatch.setenv("FDT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEVICE_HMAC_KEY", "ab" * 32)
    monkeypatch.setenv("ADMIN_USERNAME", ADMIN[0])
    monkeypatch.setenv("ADMIN_PASSWORD", ADMIN[1])
    if database_url:
        monkeypatch.setenv("DATABASE_URL", database_url)
    else:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    st.cache_resource.clear()
    return tmp_path


def new_app():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.run()
    assert not app.exception, app.exception
    return app


def button(app, label):
    return next(b for b in app.button if b.label == label)


def text(app, label):
    return [t for t in app.text_input if t.label == label][-1]


def login(app, username, password):
    text(app, "Username (Badge / Employee No.)").set_value(username)
    text(app, "Password").set_value(password)
    button(app, "Sign in").click().run()
    assert not app.exception, app.exception
    if signed_in(app):
        app.run()
    return app


def signed_in(app):
    return any("Signed in as" in m.value for m in app.markdown)


def test_login_required_and_lockout(env):
    app = new_app()
    assert not signed_in(app) and len(app.tabs) == 0
    login(app, ADMIN[0], "wrong-password")
    assert any("Invalid username or password" in e.value for e in app.error)
    for _ in range(4):
        login(app, ADMIN[0], "wrong-password")
    login(app, *ADMIN)
    assert any("temporarily locked" in e.value for e in app.error)
    assert not signed_in(app)


def test_first_time_setup(env, monkeypatch):
    monkeypatch.delenv("ADMIN_USERNAME")
    monkeypatch.delenv("ADMIN_PASSWORD")
    app = new_app()
    assert any("No accounts exist yet" in i.value for i in app.info)
    text(app, "Administrator username").set_value("chief")
    text(app, "Full name").set_value("Chief Admin")
    text(app, "New administrator password").set_value("Chief@2026")
    text(app, "Confirm administrator password").set_value("Chief@2026")
    button(app, "Create administrator").click().run()
    app.run()
    login(app, "chief", "Chief@2026")
    assert signed_in(app)


def test_admin_creates_officer_and_ledger_is_scoped(env):
    app = login(new_app(), *ADMIN)
    assert signed_in(app) and len(app.tabs) == 5
    text(app, "Username (Badge / Employee No.)").set_value("SI-MH-0999")
    text(app, "Full name & rank").set_value("SI Rao")
    text(app, "Temporary password (min 8 chars)").set_value("Officer@123")
    button(app, "Create account").click().run()
    assert any("Account 'SI-MH-0999' created" in s.value for s in app.success)

    app.selectbox[0].set_value("Negative (daylight)")
    button(app, "Analyse & Seal Evidence").click().run()
    assert any("Negative_Blank" in s.value for s in app.success)

    officer = login(new_app(), "SI-MH-0999", "Officer@123")
    assert signed_in(officer) and len(officer.tabs) == 4
    assert text(officer, "Operator ID (from signed-in account)").value == "SI-MH-0999"
    assert any("No records found." in i.value for i in officer.info)

    officer.selectbox[0].set_value("Positive · Amphetamine (warm/sodium light)")
    officer.text_input[1].set_value("19.0760, 72.8777")
    button(officer, "Analyse & Seal Evidence").click().run()
    assert any("Positive_Amphetamine" in e.value for e in officer.error)
    assert any("Showing 1 of 1 records" in c.value for c in officer.caption)

    app.run()
    assert any("Showing 2 of 2 records" in c.value for c in app.caption)

    next(s for s in app.selectbox if s.label == "Account").set_value("SI-MH-0999")
    next(s for s in app.selectbox if s.label == "Action").set_value("Deactivate")
    button(app, "Apply").click().run()
    officer.run()
    assert not signed_in(officer)


def test_capture_flow_duplicate_search_and_tamper(env, database_url):
    app = login(new_app(), *ADMIN)
    button(app, "Analyse & Seal Evidence").click().run()
    assert any("No image provided" in e.value for e in app.error)

    app.selectbox[0].set_value("Negative (daylight)")
    app.text_input[1].set_value("Delhi near India Gate")
    button(app, "Analyse & Seal Evidence").click().run()
    assert any("GPS location must be" in e.value for e in app.error)

    app.text_input[1].set_value("19.0760, 72.8777")
    app.selectbox[0].set_value("Positive · Amphetamine (warm/sodium light)")
    button(app, "Analyse & Seal Evidence").click().run()
    assert any("PRESUMPTIVE POSITIVE" in m.value for m in app.markdown)
    app.run()
    assert any("PRESUMPTIVE POSITIVE" in m.value for m in app.markdown)

    button(app, "Analyse & Seal Evidence").click().run()
    assert any("already sealed as record FDT-000001" in w.value for w in app.warning)

    app.selectbox[0].set_value("Negative (daylight)")
    button(app, "Analyse & Seal Evidence").click().run()
    assert any("Negative_Blank" in s.value for s in app.success)

    app.selectbox[0].set_value("Error case · occluded corner")
    button(app, "Analyse & Seal Evidence").click().run()
    assert any(e.value.startswith("Error: Reference card obscured") for e in app.error)

    search = text(app, "Search records")
    search.set_value("NEGATIVE_blank").run()
    assert any("Showing 1 of 2 records" in c.value for c in app.caption)
    search.set_value("zzz-no-match").run()
    assert any("Showing 0 of 2 records" in c.value for c in app.caption)

    button(app, "Verify entire chain").click().run()
    assert any("All 2 records verified" in s.value for s in app.success)

    button(app, "Prepare full evidence archive (ZIP)").click().run()
    assert any(b.label == "Download evidence archive" for b in app.get("download_button"))

    sql = "UPDATE test_logs SET operator_id = 'FORGED' WHERE id = 1"
    if database_url:
        import psycopg

        with psycopg.connect(database_url, autocommit=True) as conn:
            conn.execute(sql)
    else:
        import sqlite3

        with sqlite3.connect(env / "field_tests.db") as conn:
            conn.execute(sql)
    button(app, "Verify entire chain").click().run()
    assert any("records failed" in e.value for e in app.error)
