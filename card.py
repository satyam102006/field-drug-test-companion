"""Printable reference calibration card and deterministic field-capture simulator."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from pipeline import (
    CARD_SIZE,
    MARKER_ORIGINS,
    MARKER_SIZE,
    PATCH_PRINT_SIZE,
    REFERENCE_PATCHES,
    TEST_WINDOW_SIZE,
)

QUIET_ZONE = 20  # extra white border (card units) around the 400 x 400 plane


def render_reference_card(scale: int = 5, test_bgr: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """Render the card. With test_bgr the reaction window is filled (demo sample)."""
    s = scale
    size = (CARD_SIZE + 2 * QUIET_ZONE) * s
    card = np.full((size, size, 3), 255, np.uint8)
    off = QUIET_ZONE * s
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    def box(x0, y0, x1, y1):
        return (off + int(x0 * s), off + int(y0 * s)), (off + int(x1 * s), off + int(y1 * s))

    cv2.rectangle(card, (2, 2), (size - 3, size - 3), (170, 170, 170), max(1, s // 2))

    for marker_id, (x, y) in MARKER_ORIGINS.items():
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE * s)
        (px, py), _ = box(x, y, x, y)
        card[py:py + MARKER_SIZE * s, px:px + MARKER_SIZE * s] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    half = PATCH_PRINT_SIZE // 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, ((cx, cy), bgr) in REFERENCE_PATCHES.items():
        p0, p1 = box(cx - half, cy - half, cx + half, cy + half)
        cv2.rectangle(card, p0, p1, tuple(int(v) for v in bgr), -1)
        cv2.putText(card, name.upper(), (p0[0], p1[1] + 9 * s), font, 0.22 * s, (60, 60, 60), max(1, s // 3), cv2.LINE_AA)

    w = TEST_WINDOW_SIZE // 2
    c = CARD_SIZE // 2
    p0, p1 = box(c - w, c - w, c + w, c + w)
    if test_bgr is not None:
        cv2.rectangle(card, p0, p1, tuple(int(v) for v in test_bgr), -1)
    else:
        step = 6 * s
        for t in range(p0[0], p1[0], 2 * step):
            e = min(t + step, p1[0])
            cv2.line(card, (t, p0[1]), (e, p0[1]), (120, 120, 120), max(1, s // 3))
            cv2.line(card, (t, p1[1]), (e, p1[1]), (120, 120, 120), max(1, s // 3))
        for t in range(p0[1], p1[1], 2 * step):
            e = min(t + step, p1[1])
            cv2.line(card, (p0[0], t), (p0[0], e), (120, 120, 120), max(1, s // 3))
            cv2.line(card, (p1[0], t), (p1[0], e), (120, 120, 120), max(1, s // 3))
    cv2.putText(card, "TEST REACTION WINDOW", (p0[0], p0[1] - 4 * s), font, 0.2 * s, (60, 60, 60), max(1, s // 3), cv2.LINE_AA)

    title_org = (off + 72 * s, off + 30 * s)
    cv2.putText(card, "FIELD TEST CALIBRATION CARD", title_org, font, 0.3 * s, (80, 40, 10), max(1, s // 2), cv2.LINE_AA)
    cv2.putText(card, "PS ID26231 | ArUco 4x4_50 | IDs 0-3", (off + 90 * s, off + 45 * s), font, 0.2 * s,
                (90, 90, 90), max(1, s // 3), cv2.LINE_AA)
    cv2.putText(card, "Print at 100% scale. Keep flat, matte, unobstructed.", (off + 70 * s, off + 372 * s), font,
                0.18 * s, (90, 90, 90), max(1, s // 3), cv2.LINE_AA)
    return card


def simulate_capture(
    card_bgr: np.ndarray,
    channel_gains: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    tilt: float = 0.12,
    seed: int = 7,
    occlude_corner: bool = False,
    glare: bool = False,
    canvas: Tuple[int, int] = (1280, 960),
) -> np.ndarray:
    """Project the card onto a cluttered background with perspective, colour cast and sensor noise."""
    rng = np.random.default_rng(seed)
    W, Hh = canvas
    bg = np.full((Hh, W, 3), (70, 85, 95), np.float32)
    bg += rng.normal(0, 12, bg.shape).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), 6)

    h, w = card_bgr.shape[:2]
    side = min(W, Hh) * 0.75
    cx, cy = W / 2, Hh / 2
    j = side * tilt
    dst = np.float32([
        [cx - side / 2 + rng.uniform(-j, j), cy - side / 2 + rng.uniform(-j, j)],
        [cx + side / 2 + rng.uniform(-j, j), cy - side / 2 + rng.uniform(-j, j)],
        [cx + side / 2 + rng.uniform(-j, j), cy + side / 2 + rng.uniform(-j, j)],
        [cx - side / 2 + rng.uniform(-j, j), cy + side / 2 + rng.uniform(-j, j)],
    ])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    P = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card_bgr.astype(np.float32), P, (W, Hh))
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), P, (W, Hh))[..., None]
    img = bg * (1 - mask) + warped * mask

    img = img * np.array(channel_gains, np.float32)
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)

    if glare:
        yy, xx = np.mgrid[0:Hh, 0:W]
        spot = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (side * 0.06) ** 2)))
        img += (spot * 400)[..., None]
    if occlude_corner:
        tl = dst[0]
        cv2.circle(img, (int(tl[0] + side * 0.08), int(tl[1] + side * 0.08)), int(side * 0.14), (30, 40, 55), -1)

    return np.clip(img, 0, 255).astype(np.uint8)
