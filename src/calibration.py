import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# Court template size in pixels (940 x 500 ≈ 94ft x 50ft at 10 px/ft)
COURT_W = 940
COURT_H = 500

# Calibration presets: user clicks these landmarks in order on the broadcast frame.
# court_pts are the corresponding known positions on the top-down template.
CALIBRATION_PRESETS = {
    "4corner": {
        "labels": [
            "Top-left corner of court",
            "Top-right corner of court",
            "Bottom-right corner of court",
            "Bottom-left corner of court",
        ],
        "court_pts": [
            (0, 0),
            (COURT_W, 0),
            (COURT_W, COURT_H),
            (0, COURT_H),
        ],
    },
    "6point": {
        "labels": [
            "Top-left corner of court",
            "Top-right corner of court",
            "Bottom-right corner of court",
            "Bottom-left corner of court",
            "Half-court line top",
            "Half-court line bottom",
        ],
        "court_pts": [
            (0, 0),
            (COURT_W, 0),
            (COURT_W, COURT_H),
            (0, COURT_H),
            (COURT_W // 2, 0),
            (COURT_W // 2, COURT_H),
        ],
    },
}


def run_calibration(
    frame: np.ndarray,
    preset: str = "4corner",
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, dict]:
    """Interactive calibration UI: user clicks court landmarks on the frame.

    Controls:
        Left-click  — place next point
        r           — reset all points
        c           — confirm (only when all points placed)
        q           — quit / cancel

    Returns:
        H          — 3×3 homography matrix (image space → court space)
        calib_data — dict containing points and matrix (also saved to disk)
    """
    config = CALIBRATION_PRESETS[preset]
    labels: List[str] = config["labels"]
    court_pts: List[Tuple[int, int]] = config["court_pts"]
    n_pts = len(labels)

    image_pts: List[Tuple[int, int]] = []
    display = frame.copy()
    win = "Calibration  |  click points in order  |  r=reset  c=confirm  q=quit"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(image_pts) < n_pts:
            image_pts.append((x, y))
            _redraw(display, frame, image_pts, labels)
            cv2.imshow(win, display)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1280), min(frame.shape[0], 720))
    cv2.setMouseCallback(win, on_mouse)
    _redraw(display, frame, image_pts, labels)
    cv2.imshow(win, display)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            image_pts.clear()
            _redraw(display, frame, image_pts, labels)
            cv2.imshow(win, display)
        elif key == ord("c"):
            if len(image_pts) == n_pts:
                break
            else:
                print(f"[calibration] Need {n_pts} points, have {len(image_pts)}. Keep clicking.")
        elif key == ord("q"):
            cv2.destroyWindow(win)
            raise RuntimeError("Calibration cancelled by user.")

    cv2.destroyWindow(win)

    src = np.array(image_pts, dtype=np.float32)
    dst = np.array(court_pts, dtype=np.float32)
    H, status = cv2.findHomography(src, dst)

    if H is None:
        raise RuntimeError("Homography computation failed. Try different / less collinear points.")

    inliers = int(status.sum()) if status is not None else n_pts
    print(f"[calibration] Homography computed ({inliers}/{n_pts} inliers)")

    calib_data = {
        "preset": preset,
        "image_pts": [list(p) for p in image_pts],
        "court_pts": [list(p) for p in court_pts],
        "H": H.tolist(),
        "court_w": COURT_W,
        "court_h": COURT_H,
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(calib_data, f, indent=2)
        print(f"[calibration] Saved → {save_path}")

    return H, calib_data


def load_calibration(path: str) -> Tuple[np.ndarray, dict]:
    """Load a previously saved calibration JSON."""
    with open(path) as f:
        data = json.load(f)
    H = np.array(data["H"], dtype=np.float64)
    print(f"[calibration] Loaded from {path}")
    return H, data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _redraw(
    display: np.ndarray,
    original: np.ndarray,
    pts: List[Tuple[int, int]],
    labels: List[str],
) -> None:
    display[:] = original
    for i, (x, y) in enumerate(pts):
        cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(display, (x, y), 9, (0, 0, 0), 1)
        cv2.putText(display, str(i + 1), (x + 11, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    n = len(pts)
    if n < len(labels):
        msg = f"Point {n + 1}/{len(labels)}: {labels[n]}"
        color = (0, 200, 255)
    else:
        msg = "All points placed. Press 'c' to confirm or 'r' to reset."
        color = (0, 255, 120)

    # Semi-transparent banner
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)
    cv2.putText(display, msg, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display, "r = reset   c = confirm   q = quit", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
