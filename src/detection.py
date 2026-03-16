import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── YOLO-pose keypoint indices (COCO 17-point skeleton) ──────────────────────
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12
KP_LEFT_ANKLE     = 15
KP_RIGHT_ANKLE    = 16
KP_CONF_THRESH    = 0.3   # minimum keypoint confidence to trust


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    anchor: Tuple[float, float]  # ground-contact point in image space
    keypoints: Optional[np.ndarray] = None  # (17, 3): x, y, conf per keypoint

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def _ankle_anchor(
    kps: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
) -> Tuple[float, float]:
    """Return ankle midpoint if both ankles are visible; fall back to bbox bottom-center."""
    valid = []
    for idx in (KP_LEFT_ANKLE, KP_RIGHT_ANKLE):
        if kps[idx, 2] > KP_CONF_THRESH:
            valid.append((float(kps[idx, 0]), float(kps[idx, 1])))
    if valid:
        return (
            sum(p[0] for p in valid) / len(valid),
            sum(p[1] for p in valid) / len(valid),
        )
    # Fallback: bottom-center of bounding box
    return ((x1 + x2) / 2.0, y2)


# ── Player detector (pose) ────────────────────────────────────────────────────

class PlayerDetector:
    """
    YOLOv8-pose person detector.
    Uses ankle midpoints as the court-contact anchor for more accurate projection.
    Weights are downloaded automatically on first use (~7 MB).
    """

    def __init__(self, confidence_threshold: float = 0.4):
        self.conf = confidence_threshold
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError(
                    "ultralytics is required.\n"
                    "Install with: pip install ultralytics"
                )
            self._model = YOLO("yolov8n-pose.pt")
            print("[detection] YOLOv8n-pose loaded")
        return self._model

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run pose estimation on a single BGR frame; return player detections."""
        model = self._load_model()
        results = model(frame, conf=self.conf, classes=[0], verbose=False)

        detections: List[Detection] = []
        for r in results:
            n = len(r.boxes)
            if n == 0:
                continue

            kps_data = (
                r.keypoints.data.cpu().numpy()  # (N, 17, 3)
                if r.keypoints is not None and r.keypoints.data is not None
                else None
            )

            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                kps = kps_data[i] if kps_data is not None and i < len(kps_data) else None
                anchor = _ankle_anchor(kps, x1, y1, x2, y2) if kps is not None \
                    else ((x1 + x2) / 2.0, y2)

                detections.append(Detection(x1, y1, x2, y2, conf, anchor, keypoints=kps))

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        all_detections: List[List[Detection]] = []
        n = len(frames)
        for i, frame in enumerate(frames):
            if i % 20 == 0:
                print(f"[detection] Frame {i}/{n} ...")
            all_detections.append(self.detect(frame))
        print(f"[detection] Done — {n} frames.")
        return all_detections


# ── Ball detector ─────────────────────────────────────────────────────────────

class BallDetector:
    """
    Detects basketballs using YOLOv8n (COCO class 32 = sports ball).
    Returns the image-space center of the most confident detection per frame.
    """

    def __init__(self, confidence_threshold: float = 0.25):
        self.conf = confidence_threshold
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")
            print("[ball] YOLOv8n loaded for ball detection")
        return self._model

    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Return (x, y) center of the most confident ball, or None."""
        model = self._load_model()
        results = model(frame, conf=self.conf, classes=[32], verbose=False)

        best_pos: Optional[Tuple[float, float]] = None
        best_conf = 0.0

        for r in results:
            for box in r.boxes:
                c = float(box.conf[0])
                if c > best_conf:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    best_conf = c

        return best_pos

    def detect_batch(self, frames: List[np.ndarray]) -> List[Optional[Tuple[float, float]]]:
        results: List[Optional[Tuple[float, float]]] = []
        n = len(frames)
        for i, frame in enumerate(frames):
            if i % 20 == 0:
                print(f"[ball] Frame {i}/{n} ...")
            results.append(self.detect(frame))
        found = sum(1 for r in results if r is not None)
        print(f"[ball] Done — ball detected in {found}/{n} frames.")
        return results
