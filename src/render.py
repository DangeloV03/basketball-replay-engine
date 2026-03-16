from __future__ import annotations

import cv2
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.tracking import Track

# Court template dimensions (matches calibration.py)
COURT_W = 940
COURT_H = 500

# ── YOLO-pose skeleton ────────────────────────────────────────────────────────
# Each tuple is (kp_index_a, kp_index_b, bgr_color)
_SKELETON: List[Tuple[int, int, Tuple[int, int, int]]] = [
    # Torso
    (5,  6,  (60, 200, 60)),   # shoulders
    (5,  11, (60, 200, 60)),   # left side
    (6,  12, (60, 200, 60)),   # right side
    (11, 12, (60, 200, 60)),   # hips
    # Arms
    (5,  7,  (255, 140, 40)),  # left upper arm
    (7,  9,  (255, 180, 80)),  # left forearm
    (6,  8,  (40, 140, 255)),  # right upper arm
    (8,  10, (80, 180, 255)),  # right forearm
    # Legs
    (11, 13, (200, 60, 200)),  # left thigh
    (13, 15, (220, 100, 220)), # left shin
    (12, 14, (60, 200, 200)),  # right thigh
    (14, 16, (100, 220, 220)), # right shin
]
_KP_CONF = 0.3  # minimum keypoint confidence to draw


# ── Color helpers ─────────────────────────────────────────────────────────────

# Fallback track-ID colours (used when team classifier is not active)
_FALLBACK_COLORS: List[Tuple[int, int, int]] = [
    (60, 120, 255), (60, 220, 60), (255, 80, 80),
    (255, 200, 40), (200, 60, 255), (40, 220, 220),
    (255, 140, 40), (180, 40, 180), (40, 180, 120),
]

BALL_COLOR = (30, 140, 255)       # orange — ball marker (BGR)
BALL_COURT_COLOR = (20, 120, 230) # slightly darker for court


def _player_color(
    track_id: int,
    team_map: Optional[Dict[int, Tuple[int, int, int]]],
) -> Tuple[int, int, int]:
    if team_map and track_id in team_map:
        return team_map[track_id]
    return _FALLBACK_COLORS[track_id % len(_FALLBACK_COLORS)]


# ── Court drawing ─────────────────────────────────────────────────────────────

def draw_court(width: int = COURT_W, height: int = COURT_H) -> np.ndarray:
    """Draw a simplified top-down basketball court diagram."""
    court = np.full((height, width, 3), (200, 220, 180), dtype=np.uint8)

    lc = (40, 40, 40)
    lw = 2
    cy = height // 2
    mid_x = width // 2

    cv2.rectangle(court, (1, 1), (width - 2, height - 2), lc, lw)
    cv2.line(court, (mid_x, 0), (mid_x, height), lc, lw)
    cv2.circle(court, (mid_x, cy), 60, lc, lw)

    paint_w, paint_h = 190, 160

    # Left side
    cv2.rectangle(court, (0, cy - paint_h // 2), (paint_w, cy + paint_h // 2), lc, lw)
    cv2.ellipse(court, (paint_w, cy), (60, 60), 0, -90, 90, lc, lw)
    cv2.ellipse(court, (paint_w, cy), (60, 60), 0, 90, 270, lc, 1)
    cv2.ellipse(court, (0, cy), (238, 238), 0, -72, 72, lc, lw)
    cv2.line(court, (0, cy - 140), (47, cy - 140), lc, lw)
    cv2.line(court, (0, cy + 140), (47, cy + 140), lc, lw)
    cv2.circle(court, (52, cy), 9, (30, 30, 180), -1)
    cv2.circle(court, (52, cy), 9, lc, 1)
    cv2.line(court, (5, cy - 22), (5, cy + 22), lc, 3)

    # Right side (mirror)
    r_x = width - paint_w
    cv2.rectangle(court, (r_x, cy - paint_h // 2), (width, cy + paint_h // 2), lc, lw)
    cv2.ellipse(court, (r_x, cy), (60, 60), 0, 90, 270, lc, lw)
    cv2.ellipse(court, (r_x, cy), (60, 60), 0, -90, 90, lc, 1)
    cv2.ellipse(court, (width, cy), (238, 238), 0, 108, 252, lc, lw)
    cv2.line(court, (width - 47, cy - 140), (width, cy - 140), lc, lw)
    cv2.line(court, (width - 47, cy + 140), (width, cy + 140), lc, lw)
    cv2.circle(court, (width - 52, cy), 9, (30, 30, 180), -1)
    cv2.circle(court, (width - 52, cy), 9, lc, 1)
    cv2.line(court, (width - 5, cy - 22), (width - 5, cy + 22), lc, 3)

    return court


# ── Top-down replay frame ─────────────────────────────────────────────────────

def render_topdown_frame(
    court_template: np.ndarray,
    frame_positions: Dict[int, Tuple[float, float]],
    tracks: Dict[int, "Track"],
    frame_idx: int,
    team_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    jersey_map: Optional[Dict[int, int]] = None,
    ball_court_pos: Optional[Tuple[float, float]] = None,
    trail_length: int = 15,
    marker_radius: int = 8,
    show_ids: bool = True,
    show_trails: bool = True,
) -> np.ndarray:
    """Render one top-down frame onto a copy of the court template."""
    canvas = court_template.copy()

    # ── Players ───────────────────────────────────────────────────────────────
    for tid, (cx, cy) in frame_positions.items():
        color = _player_color(tid, team_map)
        ix, iy = int(round(cx)), int(round(cy))

        # Trail
        if show_trails and tid in tracks:
            hist = [p for p in tracks[tid].court_history[-trail_length:] if p is not None]
            for i in range(1, len(hist)):
                p1 = (int(round(hist[i - 1][0])), int(round(hist[i - 1][1])))
                p2 = (int(round(hist[i][0])),     int(round(hist[i][1])))
                alpha = i / max(len(hist), 1)
                tc = tuple(int(c * alpha * 0.8) for c in color)
                cv2.line(canvas, p1, p2, tc, 2, cv2.LINE_AA)

        # Player dot
        cv2.circle(canvas, (ix, iy), marker_radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (ix, iy), marker_radius, (0, 0, 0), 1, cv2.LINE_AA)

        # Label: jersey number if known, otherwise track ID
        if show_ids:
            label = f"#{jersey_map[tid]}" if (jersey_map and tid in jersey_map) else str(tid)
            tx, ty = ix + marker_radius + 3, iy - marker_radius + 4
            cv2.putText(canvas, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(canvas, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ── Ball ──────────────────────────────────────────────────────────────────
    if ball_court_pos is not None:
        bx, by = int(round(ball_court_pos[0])), int(round(ball_court_pos[1]))
        cv2.circle(canvas, (bx, by), 7, BALL_COURT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, (bx, by), 7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "B", (bx + 9, by + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(canvas, "B", (bx + 9, by + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BALL_COURT_COLOR, 1)

    # ── Legend ────────────────────────────────────────────────────────────────
    if team_map:
        from src.team_classifier import TEAM_NAMES, TEAM_COLORS_BGR
        seen_labels = set(team_map.values())
        # Deduplicate label → colour mapping for legend
        legend_items: Dict[int, Tuple[int, int, int]] = {}
        for tid, col in team_map.items():
            label_idx = _label_from_color(col)
            if label_idx is not None and label_idx in seen_labels:
                legend_items[label_idx] = col
        y0 = canvas.shape[0] - 12
        x0 = 8
        for lbl, col in sorted(legend_items.items()):
            cv2.circle(canvas, (x0 + 5, y0), 5, col, -1)
            cv2.putText(canvas, TEAM_NAMES[lbl], (x0 + 13, y0 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)
            x0 += 80

    # Frame counter
    cv2.putText(canvas, f"frame {frame_idx}", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)

    return canvas


def _label_from_color(
    color: Tuple[int, int, int],
) -> Optional[int]:
    from src.team_classifier import TEAM_COLORS_BGR
    for k, v in TEAM_COLORS_BGR.items():
        if v == color:
            return k
    return None


# ── Overlay frame ─────────────────────────────────────────────────────────────

def render_overlay_frame(
    frame: np.ndarray,
    matched_tracks: Dict[int, object],
    team_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    jersey_map: Optional[Dict[int, int]] = None,
    ball_image_pos: Optional[Tuple[float, float]] = None,
    show_ids: bool = True,
    show_skeleton: bool = True,
) -> np.ndarray:
    """Draw skeleton, team-coloured bounding boxes, jersey numbers, and ball."""
    canvas = frame.copy()

    for tid, det in matched_tracks.items():
        color = _player_color(tid, team_map)
        x1, y1 = int(det.x1), int(det.y1)
        x2, y2 = int(det.x2), int(det.y2)
        ax, ay = int(det.anchor[0]), int(det.anchor[1])

        # ── Skeleton ──────────────────────────────────────────────────────────
        if show_skeleton and det.keypoints is not None:
            kps = det.keypoints  # (17, 3)
            for kp_a, kp_b, skel_color in _SKELETON:
                if kps[kp_a, 2] > _KP_CONF and kps[kp_b, 2] > _KP_CONF:
                    p1 = (int(kps[kp_a, 0]), int(kps[kp_a, 1]))
                    p2 = (int(kps[kp_b, 0]), int(kps[kp_b, 1]))
                    cv2.line(canvas, p1, p2, skel_color, 2, cv2.LINE_AA)

            # Keypoint dots
            for idx in range(min(17, len(kps))):
                if kps[idx, 2] > _KP_CONF:
                    kx, ky = int(kps[idx, 0]), int(kps[idx, 1])
                    cv2.circle(canvas, (kx, ky), 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (kx, ky), 3, color, 1, cv2.LINE_AA)

        # ── Bounding box ──────────────────────────────────────────────────────
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # ── Anchor point (ankle) ──────────────────────────────────────────────
        cv2.circle(canvas, (ax, ay), 5, (0, 240, 240), -1)
        cv2.circle(canvas, (ax, ay), 5, (0, 0, 0), 1)

        # ── Label: jersey number or track ID ──────────────────────────────────
        if show_ids:
            jersey = jersey_map.get(tid) if jersey_map else None
            if jersey is not None:
                label = f"#{jersey}"
            else:
                label = f"T{tid}"

            lx, ly = x1, max(y1 - 6, 14)
            cv2.putText(canvas, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(canvas, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # ── Ball ──────────────────────────────────────────────────────────────────
    if ball_image_pos is not None:
        bx, by = int(ball_image_pos[0]), int(ball_image_pos[1])
        cv2.circle(canvas, (bx, by), 12, BALL_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, "ball", (bx + 14, by + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(canvas, "ball", (bx + 14, by + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 2)

    return canvas
