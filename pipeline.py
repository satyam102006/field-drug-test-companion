"""
Forensic engine for the Digital Companion for Field Drug Testing (SIH 2026, PS ID26231).

Fully deterministic pipeline (no AI/ML):
    1. Planar homography from 4 ArUco markers (DICT_4X4_50)      -> geometric normalisation
    2. Ordinary Least Squares colour-correction matrix (float32)  -> lighting calibration
    3. sRGB -> CIE XYZ (D65) -> CIE L*a*b* (float32)               -> perceptual colour space
    4. Euclidean Delta E (CIE76) against chemical anchors          -> classification
    5. SHA-256 seal + HMAC-SHA256 signature, hash-chained ledger   -> tamper evidence
    6. SQLite (WAL, busy timeout, retry with back-off)             -> searchable evidence log

The output is a PRESUMPTIVE field-test result and does not replace laboratory
confirmatory testing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from storage import GENESIS_HASH, SQLiteStorage, Storage, StorageBusyError  # noqa: F401

# ---------------------------------------------------------------------------
# Reference card geometry (units = pixels of the 400 x 400 normalised plane)
# ---------------------------------------------------------------------------
CARD_SIZE = 400
MARKER_SIZE = 50
MARKER_ORIGINS: Dict[int, Tuple[int, int]] = {   # marker id -> top-left (x, y)
    0: (10, 10),                                 # top-left
    1: (CARD_SIZE - 10 - MARKER_SIZE, 10),       # top-right
    2: (CARD_SIZE - 10 - MARKER_SIZE, CARD_SIZE - 10 - MARKER_SIZE),  # bottom-right
    3: (10, CARD_SIZE - 10 - MARKER_SIZE),       # bottom-left
}

PATCH_PRINT_SIZE = 60     # printed square side
PATCH_SAMPLE_SIZE = 30    # inner square actually sampled (avoids print/warp edges)

# name -> ((x, y) centre, ground-truth BGR)
REFERENCE_PATCHES: Dict[str, Tuple[Tuple[int, int], Tuple[int, int, int]]] = {
    "Grey":  ((100, 100), (200, 200, 200)),
    "Black": ((100, 300), (0, 0, 0)),
    "Red":   ((300, 100), (40, 40, 215)),
    "Green": ((300, 300), (40, 215, 40)),
}

TEST_WINDOW_SIZE = 100    # printed outline of the reaction window
ROI_SIZE = 40             # centre ROI sampled for classification

# Chemical anchors (BGR, as specified by the colour chart of the kit)
CHEMICAL_ANCHORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Positive_MDMA": (112, 25, 25),
    "Positive_Amphetamine": (0, 69, 255),
    "Negative_Blank": (224, 255, 255),
}
DELTA_E_THRESHOLD = 35.0
INCONCLUSIVE = "Inconclusive"

# D65 sRGB -> XYZ matrix and reference white
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

# Quality-gate thresholds
MIN_IMAGE_SIDE = 240
MIN_GLOBAL_STD = 4.0            # below this the frame is blank / corrupt
MIN_MEAN_LUMA = 18.0            # severely under-exposed
MAX_MEAN_LUMA = 245.0           # severely over-exposed
MAX_PATCH_STD = 22.0            # reference patch must be uniform
MAX_ROI_STD = 28.0              # reaction zone must be uniform (glare / shadow check)
GLARE_LEVEL = 250               # channel value considered clipped
MAX_ROI_GLARE_FRACTION = 0.20   # fraction of fully clipped pixels tolerated in ROI
MAX_CALIBRATION_RMSE = 45.0     # residual of S.M vs R (0-255 scale)
MAX_CONDITION_NUMBER = 1e6      # conditioning of S^T S

MARKER_ERROR = "Error: Reference card obscured. Please ensure all 4 corners are visible."


def load_or_create_key(path: Path) -> bytes:
    """Local-only device key. Hosted deployments must supply DEVICE_HMAC_KEY instead."""
    path = Path(path).resolve()
    if path.exists():
        key = path.read_bytes()
        if len(key) >= 32:
            return key
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


class FieldTestPipeline:
    """Deterministic colorimetric analysis and tamper-evident evidence logging."""

    def __init__(
        self,
        storage: Optional[Storage] = None,
        device_key: Optional[bytes] = None,
        db_path: str = "field_tests.db",
        key_path: str = ".device_key",
    ) -> None:
        self.storage = storage if storage is not None else SQLiteStorage(db_path)
        self._device_key = device_key if device_key is not None else load_or_create_key(Path(key_path))
        if len(self._device_key) < 32:
            raise ValueError("Device HMAC key must be at least 32 bytes")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMax = 53
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        self._anchor_lab = {
            name: self.bgr_to_lab(np.array([[bgr]], dtype=np.float32) / 255.0).reshape(3)
            for name, bgr in CHEMICAL_ANCHORS_BGR.items()
        }

    # ------------------------------------------------------------------
    # Colour science
    # ------------------------------------------------------------------
    @staticmethod
    def bgr_to_lab(bgr01: np.ndarray) -> np.ndarray:
        """Exact CIE L*a*b* (D65) from BGR in [0, 1], computed in float32 (no 8-bit quantisation)."""
        bgr01 = np.clip(bgr01.astype(np.float32), 0.0, 1.0)
        rgb = bgr01[..., ::-1]
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)
        xyz = linear @ _RGB_TO_XYZ.T
        t = xyz / _D65_WHITE
        delta = 6.0 / 29.0
        f = np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0).astype(np.float32)
        L = 116.0 * f[..., 1] - 16.0
        a = 500.0 * (f[..., 0] - f[..., 1])
        b = 200.0 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    @staticmethod
    def _quality_gate(image_bgr: Any) -> Optional[str]:
        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            return "Error: Image could not be decoded. The file may be corrupt or in an unsupported format."
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
            return "Error: Unsupported image format. A 3-channel 8-bit colour image is required."
        h, w = image_bgr.shape[:2]
        if min(h, w) < MIN_IMAGE_SIDE:
            return f"Error: Image resolution too low ({w}x{h}). Capture at least {MIN_IMAGE_SIDE}px on the short side."
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mean, std = float(gray.mean()), float(gray.std())
        if std < MIN_GLOBAL_STD:
            return "Error: Image is blank or corrupted (no usable detail). Please recapture."
        if mean < MIN_MEAN_LUMA:
            return "Error: Image severely under-exposed. Add light (torch/headlamp) and recapture."
        if mean > MAX_MEAN_LUMA:
            return "Error: Image severely over-exposed or washed out by glare. Shade the card and recapture."
        return None

    def _detect_markers(self, image_bgr: np.ndarray) -> Tuple[Dict[int, np.ndarray], List[int]]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        best: Dict[int, np.ndarray] = {}
        seen: List[int] = []
        for candidate in (gray, self._clahe.apply(gray)):
            corners, ids, _ = self.detector.detectMarkers(candidate)
            found: Dict[int, np.ndarray] = {}
            if ids is not None:
                for c, i in zip(corners, ids.flatten()):
                    i = int(i)
                    if i in MARKER_ORIGINS and i not in found:
                        found[i] = c.reshape(4, 2).astype(np.float32)
            if len(found) > len(best):
                best = found
            if len(best) == 4:
                break
        seen = sorted(best.keys())
        return best, seen

    @staticmethod
    def _card_marker_corners(marker_id: int) -> np.ndarray:
        x, y = MARKER_ORIGINS[marker_id]
        s = MARKER_SIZE
        return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float32)

    def _compute_homography(self, markers: Dict[int, np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
        src = np.concatenate([markers[i] for i in sorted(markers)], axis=0)
        dst = np.concatenate([self._card_marker_corners(i) for i in sorted(markers)], axis=0)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or not np.all(np.isfinite(H)):
            return None, float("inf")
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        reproj = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
        return H, reproj

    @staticmethod
    def _square(img: np.ndarray, centre: Tuple[int, int], size: int) -> np.ndarray:
        cx, cy = centre
        h = size // 2
        return img[cy - h: cy + h, cx - h: cx + h]

    def _calibrate(self, warped_f32: np.ndarray) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """OLS colour correction: M = (S^T S)^-1 S^T R, all in normalised float32."""
        s_rows, r_rows, patch_report = [], [], {}
        for name, (centre, truth) in REFERENCE_PATCHES.items():
            patch = self._square(warped_f32, centre, PATCH_SAMPLE_SIZE)
            std = float(patch.reshape(-1, 3).std(axis=0).max() * 255.0)
            if std > MAX_PATCH_STD:
                return None, (
                    f"Error: '{name}' reference patch is obstructed, shadowed or has glare "
                    f"(non-uniformity {std:.1f}). Remove obstructions and recapture."
                )
            measured = patch.reshape(-1, 3).mean(axis=0)
            s_rows.append(measured)
            r_rows.append(np.array(truth, dtype=np.float32) / 255.0)
            patch_report[name] = {"measured_bgr": (measured * 255.0).round(1).tolist(), "truth_bgr": list(truth)}

        grey_measured = s_rows[0] * 255.0
        if np.all(grey_measured >= GLARE_LEVEL):
            return None, "Error: Reference card is saturated by glare/flash (grey patch clipped). Tilt the card or disable flash."

        S = np.stack(s_rows).astype(np.float32)   # n x 3 captured
        R = np.stack(r_rows).astype(np.float32)   # n x 3 ground truth
        StS = S.T @ S
        cond = float(np.linalg.cond(StS.astype(np.float64)))
        if not np.isfinite(cond) or cond > MAX_CONDITION_NUMBER:
            return None, (
                "Error: Lighting calibration is ill-conditioned (reference colours indistinguishable). "
                "Image may be corrupted, monochrome or under extreme coloured light."
            )
        M = (np.linalg.inv(StS) @ S.T @ R).astype(np.float32)

        rmse = float(np.sqrt(np.mean((S @ M - R) ** 2)) * 255.0)
        if rmse > MAX_CALIBRATION_RMSE:
            return None, (
                f"Error: Lighting too extreme to calibrate reliably (residual {rmse:.1f}). "
                "Move to more even light and recapture."
            )

        h, w = warped_f32.shape[:2]
        calibrated = (warped_f32.reshape(-1, 3) @ M).reshape(h, w, 3)
        calibrated = np.clip(calibrated, 0.0, 1.0).astype(np.float32)
        return {"M": M, "rmse": rmse, "cond": cond, "calibrated_f32": calibrated, "patches": patch_report}, None

    def _classify(self, roi_f32: np.ndarray) -> Tuple[str, float, np.ndarray, Dict[str, float]]:
        lab_pixels = self.bgr_to_lab(roi_f32).reshape(-1, 3)
        lab = lab_pixels.mean(axis=0)
        distances = {
            name: float(np.sqrt(np.sum((lab - anchor) ** 2))) for name, anchor in self._anchor_lab.items()
        }
        best = min(distances, key=distances.get)
        best_de = distances[best]
        verdict = best if best_de < DELTA_E_THRESHOLD else INCONCLUSIVE
        return verdict, best_de, lab, distances

    # ------------------------------------------------------------------
    # Cryptographic seal
    # ------------------------------------------------------------------
    @staticmethod
    def fingerprint(image_bgr: np.ndarray) -> Optional[str]:
        """SHA-256 of the lossless PNG encoding of the captured pixels."""
        ok, png = cv2.imencode(".png", image_bgr)
        return hashlib.sha256(png.tobytes()).hexdigest() if ok else None

    @staticmethod
    def compute_seal(
        image_png: bytes,
        timestamp: str,
        gps: str,
        operator_id: str,
        result: str,
        delta_e: float,
        raw_sha256: str,
        prev_hash: str,
    ) -> str:
        """h = SHA-256(I_cal || T || G || O_id || R || dE || H_raw || H_prev)."""
        h = hashlib.sha256()
        h.update(image_png)
        for field in (timestamp, gps, operator_id, result, f"{delta_e:.4f}", raw_sha256, prev_hash):
            h.update(b"\x1f")
            h.update(field.encode("utf-8"))
        return h.hexdigest()

    def _sign(self, seal: str) -> str:
        return hmac.new(self._device_key, seal.encode("ascii"), hashlib.sha256).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_image(
        self, image_bgr: np.ndarray, operator_id: str, gps_coords: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            return self._process(image_bgr, operator_id, gps_coords)
        except StorageBusyError as exc:
            return None, f"Error: Evidence database is busy or unreachable ({exc}). Record NOT saved - please retry."
        except Exception as exc:
            if _is_db_error(exc):
                return None, f"Error: Evidence database failure ({exc}). Record NOT saved."
            if isinstance(exc, cv2.error):
                return None, f"Error: Image processing failed - the image may be corrupted ({exc})."
            if isinstance(exc, (MemoryError, ValueError, np.linalg.LinAlgError)):
                return None, f"Error: Analysis failed ({type(exc).__name__}: {exc}). Please recapture."
            raise

    def _process(self, image_bgr, operator_id, gps_coords):
        operator_id = (operator_id or "").strip()
        gps_coords = (gps_coords or "").strip()
        if not operator_id:
            return None, "Error: Operator ID is required for chain of custody."
        if not gps_coords:
            return None, "Error: GPS location is required for chain of custody."

        err = self._quality_gate(image_bgr)
        if err:
            return None, err

        raw_sha256 = self.fingerprint(image_bgr)
        if raw_sha256 is None:
            return None, "Error: Could not encode captured image for hashing."

        # Step 1: homography
        markers, seen = self._detect_markers(image_bgr)
        if len(markers) < 4:
            detail = f" (Detected {len(seen)}/4 markers" + (f": IDs {', '.join(map(str, seen))})" if seen else ")")
            return None, MARKER_ERROR + detail
        H, reproj = self._compute_homography(markers)
        if H is None or reproj > 4.0:
            return None, "Error: Reference card geometry inconsistent (card bent, folded or partially covered). Flatten the card and recapture."
        warped = cv2.warpPerspective(image_bgr, H, (CARD_SIZE, CARD_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped_f32 = warped.astype(np.float32) / 255.0

        # Step 2: OLS calibration
        cal, err = self._calibrate(warped_f32)
        if err:
            return None, err
        calibrated_f32 = cal["calibrated_f32"]
        calibrated_u8 = np.round(calibrated_f32 * 255.0).astype(np.uint8)

        # Step 3: ROI + quality checks
        centre = (CARD_SIZE // 2, CARD_SIZE // 2)
        raw_roi = self._square(warped, centre, ROI_SIZE)
        glare_fraction = float(np.mean(np.all(raw_roi >= GLARE_LEVEL, axis=2)))
        if glare_fraction > MAX_ROI_GLARE_FRACTION:
            return None, f"Error: Specular glare on the reaction zone ({glare_fraction:.0%} clipped pixels). Tilt the device and recapture."
        roi_f32 = self._square(calibrated_f32, centre, ROI_SIZE)
        roi_std = float(cv2.cvtColor(roi_f32, cv2.COLOR_BGR2GRAY).std() * 255.0)
        if roi_std > MAX_ROI_STD:
            return None, (
                f"Error: Reaction zone is not uniform (variation {roi_std:.1f}) - likely glare, shadow or the "
                "test well is misaligned with the window. Recapture."
            )
        roi_u8 = np.round(roi_f32 * 255.0).astype(np.uint8)

        # Step 4: Delta E classification
        verdict, delta_e, lab, distances = self._classify(roi_f32)
        delta_e = round(delta_e, 4)

        # Step 5 + 6: seal and commit atomically (hash chain needs the previous seal)
        ok, cal_png = cv2.imencode(".png", calibrated_u8)
        if not ok:
            return None, "Error: Could not encode calibrated image for sealing."
        cal_bytes = cal_png.tobytes()
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        lab_json = json.dumps([round(float(v), 3) for v in lab])

        def _build(prev_hash: str) -> Dict[str, Any]:
            seal = self.compute_seal(cal_bytes, timestamp, gps_coords, operator_id, verdict, delta_e, raw_sha256, prev_hash)
            return {
                "timestamp": timestamp, "operator_id": operator_id, "gps_location": gps_coords, "result": verdict,
                "delta_e": delta_e, "hash_signature": seal, "hmac_signature": self._sign(seal), "prev_hash": prev_hash,
                "raw_image_sha256": raw_sha256, "lab_values": lab_json, "calibration_rmse": round(cal["rmse"], 4),
                "evidence_png": cal_bytes,
            }

        record_id, row = self.storage.append_record(_build)
        seal, signature, prev_hash = row["hash_signature"], row["hmac_signature"], row["prev_hash"]

        return {
            "record_id": record_id,
            "timestamp": timestamp,
            "operator_id": operator_id,
            "gps_location": gps_coords,
            "verdict": verdict,
            "delta_e": delta_e,
            "distances": distances,
            "lab": [float(v) for v in lab],
            "hash": seal,
            "hmac_signature": signature,
            "prev_hash": prev_hash,
            "raw_image_sha256": raw_sha256,
            "warped_image": warped,
            "calibrated_image": calibrated_u8,
            "roi_image": roi_u8,
            "calibration_matrix": cal["M"].tolist(),
            "calibration_rmse": cal["rmse"],
            "condition_number": cal["cond"],
            "reprojection_error_px": reproj,
            "patches": cal["patches"],
            "threshold": DELTA_E_THRESHOLD,
        }, None

    # ------------------------------------------------------------------
    # Ledger
    # ------------------------------------------------------------------
    def fetch_logs(self, operator_id: Optional[str] = None) -> pd.DataFrame:
        return self.storage.fetch_logs(operator_id)

    def find_by_raw_hash(self, raw_sha256: str) -> Optional[int]:
        return self.storage.find_by_raw_hash(raw_sha256)

    def get_record(self, record_id: int) -> Optional[Dict[str, Any]]:
        return self.storage.get_record(record_id)

    def verify_record(self, record_id: int) -> Dict[str, Any]:
        """Recompute the seal from stored evidence and check signature and chain linkage."""
        rec = self.storage.get_record(record_id)
        if rec is None:
            return {"ok": False, "checks": {"Record exists": False}}
        checks: Dict[str, bool] = {"Record exists": True}
        image_bytes = rec.get("evidence_png") or b""
        checks["Evidence image present"] = len(image_bytes) > 0
        recomputed = self.compute_seal(
            image_bytes, rec["timestamp"], rec["gps_location"], rec["operator_id"], rec["result"],
            float(rec["delta_e"]), rec["raw_image_sha256"], rec["prev_hash"],
        )
        checks["SHA-256 seal matches image + metadata"] = hmac.compare_digest(recomputed, rec["hash_signature"])
        checks["HMAC device signature valid"] = hmac.compare_digest(self._sign(rec["hash_signature"]), rec["hmac_signature"])
        checks["Hash-chain link to previous record"] = rec["prev_hash"] == self.storage.prev_hash_of(record_id)
        return {"ok": all(checks.values()), "checks": checks, "record": rec}

    def verify_chain(self) -> Tuple[int, List[int]]:
        ids = self.storage.record_ids()
        failed = [i for i in ids if not self.verify_record(i)["ok"]]
        return len(ids), failed


def _is_db_error(exc: BaseException) -> bool:
    if isinstance(exc, sqlite3.Error):
        return True
    try:
        import psycopg
    except ImportError:
        return False
    return isinstance(exc, psycopg.Error)
