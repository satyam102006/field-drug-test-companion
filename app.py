"""
Digital Companion for Field Drug Testing - Streamlit frontend (SIH 2026, PS ID26231).
Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import inspect
import io
import json
import zipfile
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from card import render_reference_card, simulate_capture
from auth import ROLES, AuthService
from pipeline import CHEMICAL_ANCHORS_BGR, DELTA_E_THRESHOLD, INCONCLUSIVE, FieldTestPipeline, load_or_create_key
from storage import StorageBusyError, open_storage

st.set_page_config(
    page_title="Digital Companion for Field Drug Testing | Government of India",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def config(name: str) -> str | None:
    """Environment variable first, then .streamlit/secrets.toml / Streamlit Cloud secrets."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        if not st.secrets.load_if_toml_exists():
            return None
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None


DATA_DIR = Path(config("FDT_DATA_DIR") or Path(__file__).resolve().parent / "data")


@st.cache_resource(show_spinner=False)
def get_services(database_url: str | None, data_dir: str, key_hex: str | None) -> tuple[FieldTestPipeline, AuthService]:
    root = Path(data_dir)
    if database_url and not key_hex:
        raise RuntimeError("DEVICE_HMAC_KEY secret is required when DATABASE_URL is set "
                           "(otherwise every restart would invalidate all signatures).")
    if key_hex:
        cleaned = key_hex.strip().strip("\"'“”‘’ \t\r\n")
        if not re.fullmatch(r"[0-9a-fA-F]{64,}", cleaned):
            raise RuntimeError(
                f"DEVICE_HMAC_KEY must be at least 64 hexadecimal characters (0-9, a-f) on one line; received "
                f"{len(cleaned)} characters, of which {sum(c not in '0123456789abcdefABCDEF' for c in cleaned)} are "
                "not hex. Regenerate with: python3 -c \"import secrets; print(secrets.token_hex(32))\""
            )
        key = bytes.fromhex(cleaned)
    else:
        key = load_or_create_key(root / ".device_key")
    storage = open_storage(database_url, str(root / "field_tests.db"), legacy_evidence_dir=str(root / "evidence"))
    return FieldTestPipeline(storage=storage, device_key=key), AuthService(storage)


@st.cache_data(show_spinner=False)
def reference_card_png() -> bytes:
    return cv2.imencode(".png", render_reference_card(scale=6))[1].tobytes()


@st.cache_data(show_spinner=False)
def demo_sample_png(kind: str) -> bytes:
    presets = {
        "positive_mdma": dict(test_bgr=CHEMICAL_ANCHORS_BGR["Positive_MDMA"], gains=(1.12, 0.92, 0.62), seed=11),
        "positive_amphetamine": dict(test_bgr=CHEMICAL_ANCHORS_BGR["Positive_Amphetamine"], gains=(0.62, 0.85, 1.12), seed=5),
        "negative": dict(test_bgr=CHEMICAL_ANCHORS_BGR["Negative_Blank"], gains=(0.85, 0.95, 1.05), seed=3),
        "occluded": dict(test_bgr=CHEMICAL_ANCHORS_BGR["Positive_MDMA"], gains=(1.0, 1.0, 1.0), seed=2, occlude=True),
    }
    p = presets[kind]
    img = simulate_capture(
        render_reference_card(scale=3, test_bgr=p["test_bgr"]),
        channel_gains=p["gains"],
        seed=p["seed"],
        occlude_corner=p.get("occlude", False),
    )
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


VERDICT_META = {
    "Positive_MDMA": ("PRESUMPTIVE POSITIVE", "MDMA / Methylenedioxy-type", "positive"),
    "Positive_Amphetamine": ("PRESUMPTIVE POSITIVE", "Amphetamine-type stimulant", "positive"),
    "Negative_Blank": ("NEGATIVE", "No reaction detected", "negative"),
    INCONCLUSIVE: ("INCONCLUSIVE", f"No reference within ΔE < {DELTA_E_THRESHOLD:.0f}", "inconclusive"),
}

GPS_RE = re.compile(r"^\s*([-+]?\d{1,2}(?:\.\d+)?)\s*[, ]\s*([-+]?\d{1,3}(?:\.\d+)?)\s*$")


_IMAGE_PARAMS = inspect.signature(st.image).parameters


def show_image(image, **kwargs) -> None:
    """Full-width image on both old (use_column_width) and new (width="stretch") Streamlit."""
    if "width" not in kwargs:
        if "width" in _IMAGE_PARAMS and isinstance(_IMAGE_PARAMS["width"].default, str):
            kwargs["width"] = "stretch"
        elif "use_container_width" in _IMAGE_PARAMS:
            kwargs["use_container_width"] = True
        elif "use_column_width" in _IMAGE_PARAMS:
            kwargs["use_column_width"] = True
    st.image(image, **kwargs)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;700&display=swap');
:root { --navy:#0b2e59; --navy2:#123f78; --saffron:#ff9933; --igreen:#138808; --ink:#1f2937; --line:#d8dee9; }
html, body, [class*="css"], .stMarkdown, .stTextInput, button { font-family:'Noto Sans', 'Segoe UI', Arial, sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.block-container { padding-top:0 !important; padding-bottom:0 !important; max-width:1180px; }
.stApp { background:#f4f6fa; }

.gov-topbar { background:#1a1a1a; color:#e5e7eb; font-size:12.5px; padding:6px 18px; display:flex; justify-content:space-between;
              margin:0 -1rem; flex-wrap:wrap; gap:6px; }
.gov-topbar b { color:#fff; letter-spacing:.3px; }
.gov-header { background:#fff; padding:16px 18px; display:flex; align-items:center; gap:16px; margin:0 -1rem; border-bottom:1px solid var(--line); }
.gov-header .emblem { width:58px; height:58px; flex:0 0 58px; }
.gov-header h1 { font-size:24px; margin:0; color:var(--navy); font-weight:700; line-height:1.2; padding:0; }
.gov-header .sub { color:#4b5563; font-size:13.5px; margin-top:2px; }
.gov-header .dept { color:#b45309; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.6px; }
.gov-header .right { margin-left:auto; text-align:right; font-size:12px; color:#374151; }
.badge { display:inline-block; padding:3px 10px; border-radius:3px; font-size:11px; font-weight:700; letter-spacing:.5px; }
.badge-proto { background:#fff4e5; color:#b45309; border:1px solid #f5c68a; }
.tricolour { height:5px; margin:0 -1rem; background:linear-gradient(90deg,var(--saffron) 0 33.3%, #fff 33.3% 66.6%, var(--igreen) 66.6%); }
.notice { background:#fffbeb; border-left:4px solid var(--saffron); padding:9px 14px; font-size:13px; color:#78350f; margin:14px 0 6px; }

.stTabs [data-baseweb="tab-list"] { background:var(--navy); gap:0; padding:0 6px; border-radius:4px 4px 0 0; flex-wrap:wrap; }
.stTabs [data-baseweb="tab"] { color:#dbe4f3 !important; font-weight:600; padding:12px 18px; height:auto; }
.stTabs [data-baseweb="tab"] p { font-size:14.5px; }
.stTabs [aria-selected="true"] { background:var(--navy2); color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--saffron); height:3px; }
.stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs [data-baseweb="tab-panel"] { background:#fff; border:1px solid var(--line); border-top:none; padding:22px 20px; border-radius:0 0 4px 4px; }

.section-title { color:var(--navy); font-weight:700; font-size:17px; border-bottom:2px solid var(--saffron); padding-bottom:6px; margin:4px 0 14px; }
.step { background:#f8fafc; border:1px solid var(--line); border-radius:4px; padding:10px 12px; font-size:13px; height:100%; }
.step b { color:var(--navy); }

.verdict { border-radius:6px; padding:18px 20px; color:#fff; margin:6px 0 12px; }
.verdict .label { font-size:12px; letter-spacing:1px; opacity:.9; text-transform:uppercase; }
.verdict .main { font-size:28px; font-weight:800; letter-spacing:.5px; line-height:1.2; }
.verdict .detail { font-size:14.5px; opacity:.95; }
.verdict.positive { background:linear-gradient(135deg,#b91c1c,#7f1d1d); }
.verdict.negative { background:linear-gradient(135deg,#15803d,#14532d); }
.verdict.inconclusive { background:linear-gradient(135deg,#b45309,#78350f); }

.seal { background:#0f172a; color:#a7f3d0; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:13px; padding:12px 14px;
        border-radius:4px; word-break:break-all; border-left:4px solid var(--igreen); }
.seal small { color:#94a3b8; display:block; font-size:11px; margin-bottom:3px; letter-spacing:.6px; }
.cvcap { text-align:center; font-size:13px; font-weight:600; color:var(--navy); margin-top:4px; }
div[data-testid="stMetric"] { background:#f8fafc; border:1px solid var(--line); border-radius:4px; padding:10px 14px; }
div[data-testid="stMetricLabel"] p { font-weight:600; color:#475569; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background:var(--navy); border-color:var(--navy); font-weight:700; }
.stButton > button[kind="primary"]:hover { background:var(--navy2); border-color:var(--navy2); }

.userbar { font-size:13px; color:#334155; padding:9px 0 0; }
.userbar .role { background:#e0e7ff; color:#1e3a8a; font-weight:700; font-size:11px; padding:2px 7px; border-radius:3px; letter-spacing:.5px; }
.gov-footer { background:var(--navy); color:#cbd5e1; font-size:12.5px; padding:18px; margin:26px -1rem 0; line-height:1.6; }
.gov-footer b { color:#fff; }
.gov-footer .row { display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; }

@media (max-width: 640px) {
  .gov-header h1 { font-size:18px; }
  .gov-header .emblem { width:44px; height:44px; flex-basis:44px; }
  .gov-header .right, .gov-topbar .hide-sm { display:none; }
  .verdict .main { font-size:22px; }
  .stTabs [data-baseweb="tab"] { padding:10px 10px; }
  .stTabs [data-baseweb="tab-panel"] { padding:14px 10px; }
}
</style>
""",
    unsafe_allow_html=True,
)

EMBLEM_SVG = """
<svg class="emblem" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-label="Seal">
  <path d="M32 3 L56 12 V30 C56 46 45 56 32 61 C19 56 8 46 8 30 V12 Z" fill="#0b2e59"/>
  <path d="M32 8 L51 15 V30 C51 43 42 51 32 55 C22 51 13 43 13 30 V15 Z" fill="none" stroke="#ff9933" stroke-width="2"/>
  <circle cx="32" cy="31" r="11" fill="none" stroke="#fff" stroke-width="2"/>
  <g stroke="#fff" stroke-width="1.2">
    <line x1="32" y1="20" x2="32" y2="42"/><line x1="21" y1="31" x2="43" y2="31"/>
    <line x1="24.2" y1="23.2" x2="39.8" y2="38.8"/><line x1="39.8" y1="23.2" x2="24.2" y2="38.8"/>
  </g>
  <circle cx="32" cy="31" r="2.6" fill="#ff9933"/>
</svg>
"""

st.markdown(
    f"""
<div class="gov-topbar">
  <span><b>भारत सरकार</b> &nbsp;|&nbsp; <b>GOVERNMENT OF INDIA</b></span>
  <span class="hide-sm">Smart India Hackathon 2026 &nbsp;·&nbsp; Problem Statement ID26231</span>
</div>
<div class="gov-header">
  {EMBLEM_SVG}
  <div>
    <div class="dept">Narcotics Field Enforcement · Digital Evidence Initiative</div>
    <h1>Digital Companion for Field Drug Testing</h1>
    <div class="sub">Calibrated colorimetric analysis &amp; tamper-evident digital evidence record</div>
  </div>
  <div class="right">
    <span class="badge badge-proto">PROTOTYPE</span><br/>
    <span style="display:inline-block;margin-top:6px">Deterministic · No AI/ML · Tamper-evident</span>
  </div>
</div>
<div class="tricolour"></div>
<div class="notice"><b>Notice:</b> Results generated by this system are <b>presumptive field-test results</b> with a supporting
digital record. They do not replace confirmatory testing at a Forensic Science Laboratory (FSL).</div>
""",
    unsafe_allow_html=True,
)

def render_footer() -> None:
    st.markdown(
        f"""
<div class="gov-footer"><div class="row">
<span><b>Digital Companion for Field Drug Testing</b> — Prototype for Smart India Hackathon 2026 (PS ID26231).<br/>
Presumptive field results only; not a substitute for FSL confirmatory analysis.</span>
<span>Deterministic CV · OLS · CIE L*a*b* · SHA-256<br/>Last updated: {datetime.now(timezone.utc):%d %b %Y}</span>
</div></div>
""",
        unsafe_allow_html=True,
    )


try:
    with st.spinner("Connecting to the evidence database..."):
        pipeline, auth = get_services(config("DATABASE_URL"), str(DATA_DIR), config("DEVICE_HMAC_KEY"))
    auth.ensure_bootstrap_admin(config("ADMIN_USERNAME"), config("ADMIN_PASSWORD"))
except Exception as exc:  # surfaced to operator
    st.error(f"System initialisation failed: {exc}. Check the database connection settings and secrets.")
    render_footer()
    st.stop()

DB_ERRORS: tuple = (sqlite3.Error, StorageBusyError)
try:
    import psycopg

    DB_ERRORS += (psycopg.Error,)
except ImportError:
    pass


def _do_setup() -> None:
    ss = st.session_state
    if ss.get("setup_p1") != ss.get("setup_p2"):
        ss["auth_msg"] = ("error", "Passwords do not match.")
        return
    try:
        if auth.has_users():
            ss["auth_msg"] = ("error", "An administrator already exists. Please sign in.")
            return
        err = auth.create_user(ss.get("setup_u", ""), ss.get("setup_n", ""), ss.get("setup_p1", ""), "admin")
    except DB_ERRORS as exc:
        err = f"Evidence database unavailable ({exc}). Please retry."
    ss["setup_p1"] = ss["setup_p2"] = ""
    ss["auth_msg"] = ("error", err) if err else ("success", "Administrator created. Please sign in.")


def _do_login() -> None:
    ss = st.session_state
    try:
        user, msg = auth.authenticate(ss.get("login_u", ""), ss.get("login_p", ""))
    except DB_ERRORS as exc:
        user, msg = None, f"Evidence database unavailable ({exc}). Please retry."
    ss["login_p"] = ""
    if user:
        ss.pop("last", None)
        ss["user"] = user
        ss.pop("auth_msg", None)
    else:
        ss["auth_msg"] = ("error", msg)


def _do_logout() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def login_screen() -> None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        kind, msg = st.session_state.pop("auth_msg", (None, None))
        if not auth.has_users():
            st.markdown('<div class="section-title">First-time setup · create administrator</div>', unsafe_allow_html=True)
            st.info("No accounts exist yet. Create the administrator account for this installation.")
            if msg:
                st.error(msg)
            with st.form("setup_form"):
                st.text_input("Administrator username", key="setup_u")
                st.text_input("Full name", key="setup_n")
                st.text_input("New administrator password", type="password", key="setup_p1")
                st.text_input("Confirm administrator password", type="password", key="setup_p2")
                st.form_submit_button("Create administrator", type="primary", use_container_width=True, on_click=_do_setup)
            return
        st.markdown('<div class="section-title">Authorised personnel sign-in</div>', unsafe_allow_html=True)
        if msg:
            (st.success if kind == "success" else st.error)(msg)
        with st.form("login_form"):
            st.text_input("Username (Badge / Employee No.)", key="login_u")
            st.text_input("Password", type="password", key="login_p")
            st.form_submit_button("Sign in", type="primary", use_container_width=True, on_click=_do_login)
        st.caption("Accounts are issued by the unit administrator. Unauthorised access is prohibited.")


session_user = st.session_state.get("user")
if session_user:
    try:
        session_user = auth.current(session_user["username"])
    except DB_ERRORS:
        pass
    if not session_user:
        st.session_state.pop("user", None)
        st.session_state.pop("last", None)
if not session_user:
    login_screen()
    render_footer()
    st.stop()

USER = session_user
IS_ADMIN = USER["role"] == "admin"
bar_l, bar_r = st.columns([4, 1])
bar_l.markdown(
    f'<div class="userbar">Signed in as <b>{html.escape(USER["full_name"])}</b> '
    f'(<code>{html.escape(USER["username"])}</code>) · <span class="role">{USER["role"].upper()}</span> · '
    f'Storage: {html.escape(pipeline.storage.backend_name)}</div>',
    unsafe_allow_html=True,
)
bar_r.button("Sign out", use_container_width=True, on_click=_do_logout)


def decode_upload(data: bytes):
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    try:
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error:
        return None


def certificate_html(r: dict, file_sha: str) -> str:
    title, detail, _ = VERDICT_META[r["verdict"]]
    rows = [
        ("Record No.", f"FDT-{r['record_id']:06d}"),
        ("Presumptive Result", f"{title} — {detail}"),
        ("Classifier Output", r["verdict"]),
        ("ΔE (CIE76) to nearest standard", f"{r['delta_e']:.2f} (threshold {DELTA_E_THRESHOLD:.0f})"),
        ("Measured CIE L*a*b*", ", ".join(f"{v:.2f}" for v in r["lab"])),
        ("Timestamp (UTC)", r["timestamp"]),
        ("GPS Location", r["gps_location"]),
        ("Operator ID", r["operator_id"]),
        ("Captured file SHA-256", file_sha),
        ("Captured pixels SHA-256", r["raw_image_sha256"]),
        ("Digital Seal (SHA-256)", r["hash"]),
        ("Device Signature (HMAC-SHA256)", r["hmac_signature"]),
        ("Previous Record Seal (chain)", r["prev_hash"]),
    ]
    body = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Field Test Record FDT-{r['record_id']:06d}</title>
<style>body{{font-family:Arial,sans-serif;max-width:820px;margin:24px auto;color:#111}}
.bar{{height:6px;background:linear-gradient(90deg,#ff9933 0 33%,#fff 33% 66%,#138808 66%);border:1px solid #ddd}}
h1{{color:#0b2e59;font-size:20px;margin:14px 0 2px}}h2{{font-size:13px;color:#555;font-weight:normal;margin:0 0 14px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #cbd5e1;padding:7px 9px;text-align:left;vertical-align:top}}
th{{background:#eef2f7;width:34%}}td{{word-break:break-all;font-family:Menlo,Consolas,monospace}}
.note{{font-size:12px;color:#7c2d12;background:#fff7ed;border:1px solid #fed7aa;padding:8px;margin-top:14px}}
.sig{{margin-top:40px;display:flex;justify-content:space-between;font-size:12px}}</style></head><body>
<div class="bar"></div><h1>Field Drug Test — Digital Evidence Record</h1>
<h2>Government of India · Digital Companion for Field Drug Testing (Prototype, PS ID26231)</h2>
<table>{body}</table>
<div class="note">This is a presumptive field-test result and supporting digital record. It does not replace laboratory
confirmatory testing. Integrity can be verified by recomputing the SHA-256 seal from the stored calibrated image and metadata.</div>
<div class="sig"><span>Signature of Operator: ____________________</span><span>Witness: ____________________</span></div>
</body></html>"""


def render_result(r: dict, file_sha: str) -> None:
    title, detail, css = VERDICT_META[r["verdict"]]
    st.markdown(
        f"""<div class="verdict {css}"><div class="label">Record FDT-{r['record_id']:06d} · Presumptive field result</div>
        <div class="main">{title}</div><div class="detail">{html.escape(detail)}</div></div>""",
        unsafe_allow_html=True,
    )
    if css == "positive":
        st.error(f"Colour reaction matches **{r['verdict']}** (ΔE = {r['delta_e']:.2f}). Seize sample and forward to FSL for confirmation.")
    elif css == "negative":
        st.success(f"Colour reaction matches **{r['verdict']}** (ΔE = {r['delta_e']:.2f}). No presumptive indication.")
    else:
        st.warning(f"Nearest standard ΔE = {r['delta_e']:.2f} exceeds {DELTA_E_THRESHOLD:.0f}. Repeat test with a fresh kit.")

    m1, m2, m3 = st.columns(3)
    m1.metric("ΔE (CIE76)", f"{r['delta_e']:.2f}", help="Euclidean distance in L*a*b* to nearest standard; match if < 35")
    m2.metric("Calib. RMSE", f"{r['calibration_rmse']:.2f}", help="Residual of S·M vs R on the 0–255 scale")
    m3.metric("Warp error", f"{r['reprojection_error_px']:.2f}px", help="Homography reprojection RMSE")
    st.caption("Measured CIE L\\*a\\*b\\*: " + " · ".join(f"{n} = {v:.2f}" for n, v in zip(("L*", "a*", "b*"), r["lab"])))

    st.markdown(
        f"""<div class="seal"><small>SHA-256 DIGITAL SEAL</small>{r['hash']}</div>
        <div class="seal" style="margin-top:6px;border-left-color:#ff9933;color:#fde68a"><small>HMAC-SHA256 DEVICE SIGNATURE</small>{r['hmac_signature']}</div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title" style="margin-top:18px">Computer-vision progression</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        show_image(r["warped_image"], channels="BGR")
        st.markdown('<div class="cvcap">1 · Warped Matrix (homography)</div>', unsafe_allow_html=True)
    with c2:
        show_image(r["calibrated_image"], channels="BGR")
        st.markdown('<div class="cvcap">2 · Light Stripped (OLS CCM)</div>', unsafe_allow_html=True)
    with c3:
        roi_big = cv2.resize(r["roi_image"], (400, 400), interpolation=cv2.INTER_NEAREST)
        show_image(roi_big, channels="BGR")
        st.markdown('<div class="cvcap">3 · Chemical ROI (40×40)</div>', unsafe_allow_html=True)

    with st.expander("Mathematical audit trail"):
        a, b = st.columns(2)
        with a:
            st.markdown("**Colour Correction Matrix** $M = (S^TS)^{-1}S^TR$ (BGR, normalised)")
            st.dataframe(pd.DataFrame(r["calibration_matrix"], columns=["B", "G", "R"], index=["B", "G", "R"]).round(4),
                         use_container_width=True)
            st.markdown("**Reference patches** (S measured vs R ground truth)")
            st.dataframe(pd.DataFrame([{"Patch": k, "Measured BGR": v["measured_bgr"], "True BGR": v["truth_bgr"]}
                                       for k, v in r["patches"].items()]), hide_index=True, use_container_width=True)
        with b:
            st.markdown("**ΔE to each chemical standard**")
            st.dataframe(pd.DataFrame([{"Standard": k, "ΔE": round(v, 3), "Match (< 35)": v < DELTA_E_THRESHOLD}
                                       for k, v in sorted(r["distances"].items(), key=lambda kv: kv[1])]),
                         hide_index=True, use_container_width=True)
            st.markdown(f"Condition number of $S^TS$: `{r['condition_number']:.1f}`")
            st.markdown(f"Previous seal (hash chain): `{r['prev_hash'][:32]}…`")

    record = {k: r[k] for k in ("record_id", "timestamp", "operator_id", "gps_location", "verdict", "delta_e", "lab",
                                "distances", "calibration_rmse", "raw_image_sha256", "hash", "hmac_signature", "prev_hash")}
    record["captured_file_sha256"] = file_sha
    record["disclaimer"] = "Presumptive field-test result; does not replace laboratory confirmatory testing."
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download evidence certificate", certificate_html(r, file_sha), f"FDT-{r['record_id']:06d}.html",
                       "text/html", type="primary", use_container_width=True)
    d2.download_button("Download signed record (JSON)", json.dumps(record, indent=2), f"FDT-{r['record_id']:06d}.json",
                       "application/json", use_container_width=True)
    d3.download_button("Download calibrated image", cv2.imencode(".png", r["calibrated_image"])[1].tobytes(),
                       f"FDT-{r['record_id']:06d}.png", "image/png", use_container_width=True)


tab_names = ["Capture", "Evidence Ledger", "Reference Card", "Method & Help"] + (["Administration"] if IS_ADMIN else [])
tabs = st.tabs(tab_names)
tab_capture, tab_ledger, tab_card, tab_about = tabs[:4]

# ---------------------------------------------------------------------------
# Tab 1 - Capture
# ---------------------------------------------------------------------------
with tab_capture:
    use_camera = st.toggle("Use device camera", value=False, help="Off = upload a photo from gallery/files")
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="step"><b>Frame</b><br/>All 4 black corner markers fully visible.</div>', unsafe_allow_html=True)
    s2.markdown('<div class="step"><b>Light</b><br/>Avoid flash glare and hard shadows on the card.</div>', unsafe_allow_html=True)
    s3.markdown('<div class="step"><b>Align</b><br/>Reaction well centred in the test window.</div>', unsafe_allow_html=True)
    st.write("")

    with st.form("capture_form", clear_on_submit=False, border=False):
        st.markdown('<div class="section-title">Step 1 · Chain-of-custody details</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        operator_id = USER["username"]
        col_a.text_input("Operator ID (from signed-in account)", value=operator_id, disabled=True)
        gps = col_b.text_input("GPS location (latitude, longitude)", value="28.613900, 77.209000",
                               help="Decimal degrees, e.g. 28.6139, 77.2090. Copy from device GPS / maps app.")

        st.markdown('<div class="section-title">Step 2 · Capture test with reference card in frame</div>',
                    unsafe_allow_html=True)
        demo = ""
        if use_camera:
            shot = st.camera_input("Capture field test", label_visibility="collapsed")
        else:
            shot = st.file_uploader("Upload field test photograph", type=["jpg", "jpeg", "png", "bmp", "webp"])
            demo_keys = {"— none —": "", "Positive · MDMA (cool/blue light)": "positive_mdma",
                         "Positive · Amphetamine (warm/sodium light)": "positive_amphetamine",
                         "Negative (daylight)": "negative", "Error case · occluded corner": "occluded"}
            demo = demo_keys[st.selectbox("Or analyse a built-in demo capture (used only if no photo is uploaded)",
                                          list(demo_keys))]
        submitted = st.form_submit_button("Analyse & Seal Evidence", type="primary", use_container_width=True)

    def run_capture() -> None:
        data = shot.getvalue() if shot is not None else (demo_sample_png(demo) if demo else None)
        if data is None:
            st.error("No image provided. Capture or upload a photograph of the test with the reference card.")
            return
        gps_ok = GPS_RE.match(gps or "")
        if not gps_ok or abs(float(gps_ok.group(1))) > 90 or abs(float(gps_ok.group(2))) > 180:
            st.error("GPS location must be 'latitude, longitude' in decimal degrees (e.g. 28.6139, 77.2090).")
            return
        file_sha = hashlib.sha256(data).hexdigest()
        image = decode_upload(data)
        if image is None:
            st.error("Error: Image could not be decoded. The file may be corrupt or in an unsupported format.")
            return
        try:
            existing = pipeline.find_by_raw_hash(pipeline.fingerprint(image) or "")
        except DB_ERRORS as exc:
            st.error(f"Error: Evidence database is busy or unavailable ({exc}). Please retry in a few seconds.")
            return
        if existing:
            st.warning(f"This exact image is already sealed as record FDT-{existing:06d}. "
                       "Duplicate records are not permitted — see the Evidence Ledger.")
            return
        gps_norm = f"{float(gps_ok.group(1)):.6f}, {float(gps_ok.group(2)):.6f}"
        with st.spinner("Executing Deterministic Calibration..."):
            result, error = pipeline.process_image(image, operator_id, gps_norm)
        if error:
            st.session_state.pop("last", None)
            st.error(error)
            return
        st.session_state["last"] = {"file_sha": file_sha, "result": result}
        render_result(result, file_sha)

    if submitted:
        run_capture()
    elif st.session_state.get("last"):
        render_result(st.session_state["last"]["result"], st.session_state["last"]["file_sha"])

# ---------------------------------------------------------------------------
# Tab 2 - Evidence Ledger
# ---------------------------------------------------------------------------
with tab_ledger:
    st.markdown('<div class="section-title">Evidence Ledger · tamper-evident hash-chained log</div>', unsafe_allow_html=True)
    if not IS_ADMIN:
        st.caption("Showing records sealed under your account. Administrators can view the full ledger.")
    try:
        logs = pipeline.fetch_logs(None if IS_ADMIN else USER["username"])
    except Exception as exc:
        st.error(f"Could not read the evidence database (it may be locked by another process): {exc}")
        logs = None

    if logs is not None and logs.empty:
        st.info("No records found.")
    elif logs is not None:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total tests", len(logs))
        k2.metric("Positive", int(logs["result"].str.startswith("Positive").sum()))
        k3.metric("Negative", int((logs["result"] == "Negative_Blank").sum()))
        k4.metric("Inconclusive", int((logs["result"] == INCONCLUSIVE).sum()))

        query = st.text_input("Search records", placeholder="Search by operator, result, hash, GPS, date or record no.")
        view = logs.copy()
        view.insert(0, "record_no", view["id"].map(lambda i: f"FDT-{i:06d}"))
        if query.strip():
            q = query.strip().lower()
            mask = view.astype(str).apply(lambda col: col.str.lower().str.contains(q, regex=False)).any(axis=1)
            view = view[mask]
        st.caption(f"Showing {len(view)} of {len(logs)} records")
        st.dataframe(
            view,
            hide_index=True,
            use_container_width=True,
            column_order=["record_no", "timestamp", "operator_id", "gps_location", "result", "delta_e", "hash_signature"],
            column_config={
                "record_no": st.column_config.TextColumn("Record No."),
                "timestamp": st.column_config.TextColumn("Timestamp (UTC)"),
                "operator_id": st.column_config.TextColumn("Operator"),
                "gps_location": st.column_config.TextColumn("GPS"),
                "result": st.column_config.TextColumn("Result"),
                "delta_e": st.column_config.NumberColumn("ΔE", format="%.2f"),
                "hash_signature": st.column_config.TextColumn("SHA-256 Seal", width="large"),
            },
        )
        st.download_button("Export filtered ledger (CSV)", view.to_csv(index=False).encode(), "evidence_ledger.csv", "text/csv")

        st.markdown('<div class="section-title" style="margin-top:18px">Integrity verification</div>', unsafe_allow_html=True)
        v1, v2 = st.columns([2, 1])
        rid = int(v1.selectbox("Select record to verify", [f"FDT-{i:06d}" for i in logs["id"]])[4:])
        if v1.button("Verify selected record", type="primary"):
            try:
                report = pipeline.verify_record(int(rid))
            except DB_ERRORS + (OSError,) as exc:
                report = None
                st.error(f"Verification could not run (database busy or evidence unreadable): {exc}")
            if report:
                for name, passed in report["checks"].items():
                    (st.success if passed else st.error)(f"{'PASS' if passed else 'FAIL'} — {name}")
                if report["ok"]:
                    st.success(f"Record FDT-{int(rid):06d} is AUTHENTIC and unaltered.")
                else:
                    st.error(f"Record FDT-{int(rid):06d} has been TAMPERED with or its evidence is missing.")
        if IS_ADMIN and v2.button("Verify entire chain", use_container_width=True):
            try:
                with st.spinner("Recomputing every seal..."):
                    total, failed = pipeline.verify_chain()
            except DB_ERRORS + (OSError,) as exc:
                st.error(f"Verification could not run (database busy or evidence unreadable): {exc}")
            else:
                if failed:
                    st.error(f"{len(failed)} of {total} records failed: " + ", ".join(f"FDT-{i:06d}" for i in failed))
                else:
                    st.success(f"All {total} records verified. Hash chain intact.")

# ---------------------------------------------------------------------------
# Tab 3 - Reference card & demo samples
# ---------------------------------------------------------------------------
with tab_card:
    st.markdown('<div class="section-title">Printable reference calibration card</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        show_image(reference_card_png())
    with c2:
        st.markdown(
            """
- Print on **matte white paper at 100 % scale** (no "fit to page"). Laminate with matte film if possible.
- Four ArUco markers (DICT_4X4_50, IDs 0–3) give the homography anchors.
- Four calibration patches: **Grey (200,200,200)**, **Black (0,0,0)**, **Red (40,40,215)**, **Green (40,215,40)** (BGR).
- Place the test-kit reaction well / ampoule behind the dashed **test reaction window** (cut it out for vials).
"""
        )
        st.download_button("Download card (PNG, print-ready)", reference_card_png(), "reference_card_ID26231.png", "image/png",
                           type="primary", use_container_width=True)
    st.markdown('<div class="section-title" style="margin-top:18px">Demo field captures (simulated perspective &amp; lighting)</div>',
                unsafe_allow_html=True)
    samples = [("positive_mdma", "Positive · MDMA (cool/blue light)"), ("positive_amphetamine", "Positive · Amphetamine (warm light)"),
               ("negative", "Negative (daylight)"), ("occluded", "Error case · occluded corner")]
    cols = st.columns(4)
    for col, (key, label) in zip(cols, samples):
        with col:
            png = demo_sample_png(key)
            show_image(png)
            st.download_button(label, png, f"demo_{key}.jpg", "image/jpeg", use_container_width=True, key=f"dl_{key}")

# ---------------------------------------------------------------------------
# Tab 4 - Method
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown('<div class="section-title">Deterministic forensic pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        r"""
| Stage | Method | Failure handling |
|---|---|---|
| 1. Geometric normalisation | ArUco detection → RANSAC homography $H$ from 16 corner points → 400×400 plane | fewer than 4 markers, bent/folded card |
| 2. Lighting calibration | OLS colour-correction matrix $M=(S^TS)^{-1}S^TR$ in `float32` | patch glare/occlusion, ill-conditioning, high residual |
| 3. Colour space | sRGB → linear → XYZ (D65) → CIE L\*a\*b\* in `float32` | ROI glare & uniformity checks |
| 4. Classification | $\Delta E=\sqrt{\Delta L^2+\Delta a^2+\Delta b^2}$, arg-min, threshold 35 | otherwise *Inconclusive* |
| 5. Integrity | $h=\text{SHA-256}(I_{cal} \Vert T \Vert G \Vert O_{id} \Vert R \Vert \Delta E \Vert H_{raw} \Vert h_{prev})$ + HMAC | re-verifiable at any time |
| 6. Logging | SQLite WAL, `BEGIN IMMEDIATE`, busy-timeout + exponential back-off | clear "database busy" message |
"""
    )
    st.markdown(
        f"""
**Chemical reference standards (BGR):** {", ".join(f"`{k}` {v}" for k, v in CHEMICAL_ANCHORS_BGR.items())}.

**Why hash-chaining?** Each record's seal includes the previous record's seal, so deleting or editing any past entry
breaks every subsequent link — detectable with *Verify entire chain*.

**Limitations:** Results are presumptive. Anchor colours should be re-derived from the specific kit manufacturer's
colour chart before operational use. The device HMAC key (`DEVICE_HMAC_KEY` secret) must be protected; in a production
deployment it would reside in a hardware security module or the phone's Android Keystore.
"""
    )
    st.caption(f"Runtime: Streamlit {st.__version__} · Python {sys.version.split()[0]} · OpenCV {cv2.__version__} · "
               f"NumPy {np.__version__} · Storage: {pipeline.storage.backend_name}")


def build_backup_zip() -> bytes:
    buf = io.BytesIO()
    records, report = [], []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rec in pipeline.storage.iter_records():
            name = f"evidence/FDT-{rec['id']:06d}_{rec['hash_signature'][:16]}.png"
            zf.writestr(name, rec["evidence_png"])
            meta = {k: v for k, v in rec.items() if k != "evidence_png"}
            meta["evidence_file"] = name
            records.append(meta)
            check = pipeline.verify_record(rec["id"])
            report.append({"record_no": f"FDT-{rec['id']:06d}", "ok": check["ok"], "checks": check["checks"]})
        zf.writestr("records.json", json.dumps(records, indent=2, default=str))
        zf.writestr("ledger.csv", pd.DataFrame(records).to_csv(index=False))
        zf.writestr("verification_report.json", json.dumps(report, indent=2))
        zf.writestr(
            "README.txt",
            "Digital Companion for Field Drug Testing - evidence archive\n"
            f"Exported (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')} by {USER['username']}\n\n"
            "seal = SHA-256(PNG || 0x1F || timestamp || 0x1F || gps || 0x1F || operator || 0x1F || result || 0x1F\n"
            "              || delta_e(4dp) || 0x1F || raw_image_sha256 || 0x1F || prev_hash)\n"
            "Each record's prev_hash must equal the previous record's seal (genesis = 64 zeros).\n",
        )
    return buf.getvalue()


if IS_ADMIN:
    with tabs[4]:
        st.markdown('<div class="section-title">Officer accounts</div>', unsafe_allow_html=True)
        try:
            users = pipeline.storage.list_users()
        except DB_ERRORS as exc:
            users = None
            st.error(f"Could not load accounts: {exc}")
        if users is not None:
            st.dataframe(users, hide_index=True, use_container_width=True)

        a1, a2 = st.columns(2)
        with a1, st.form("create_user", clear_on_submit=True):
            st.markdown("**Create account**")
            nu = st.text_input("Username (Badge / Employee No.)")
            nn = st.text_input("Full name & rank")
            nr = st.selectbox("Role", list(ROLES))
            npw = st.text_input("Temporary password (min 8 chars)", type="password")
            if st.form_submit_button("Create account", type="primary", use_container_width=True):
                err = auth.create_user(nu, nn, npw, nr)
                (st.error(err) if err else st.success(f"Account '{nu.strip()}' created."))
        with a2, st.form("manage_user"):
            st.markdown("**Manage account**")
            names = users["username"].tolist() if users is not None else []
            target = st.selectbox("Account", names)
            action = st.selectbox("Action", ["Reset password", "Deactivate", "Reactivate"])
            rpw = st.text_input("New password (for reset)", type="password")
            if st.form_submit_button("Apply", use_container_width=True) and target:
                if action == "Reset password":
                    err = auth.set_password(target, rpw)
                    (st.error(err) if err else st.success(f"Password reset for '{target}'."))
                elif target == USER["username"]:
                    st.error("You cannot change the status of your own account.")
                else:
                    auth.set_active(target, action == "Reactivate")
                    st.success(f"'{target}' {'reactivated' if action == 'Reactivate' else 'deactivated'}.")

        st.markdown('<div class="section-title" style="margin-top:18px">Database &amp; backup</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        try:
            b1.metric("Sealed records", len(pipeline.storage.record_ids()))
        except DB_ERRORS:
            b1.metric("Sealed records", "—")
        b2.metric("Accounts", 0 if users is None else len(users))
        b3.metric("Backend", "Postgres" if "Postgres" in pipeline.storage.backend_name else "SQLite")
        st.caption("The archive contains every record, its evidence image, a CSV ledger and an integrity report. "
                   "Download it regularly and store it offline.")
        if st.button("Prepare full evidence archive (ZIP)"):
            try:
                with st.spinner("Collecting records and re-verifying every seal..."):
                    st.session_state["backup_zip"] = build_backup_zip()
            except DB_ERRORS as exc:
                st.error(f"Backup failed: {exc}")
        if st.session_state.get("backup_zip"):
            st.download_button("Download evidence archive", st.session_state["backup_zip"],
                               f"evidence_archive_{datetime.now(timezone.utc):%Y%m%d_%H%M}.zip", "application/zip",
                               type="primary")

render_footer()
