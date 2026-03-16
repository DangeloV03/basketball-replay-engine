"""
Team classifier using K-means on torso HSV color.

Clusters players into 3 groups: Team A, Team B, Referee/Other.
Uses hue + saturation (ignoring brightness) for lighting robustness.
Referee detection heuristic: lowest saturation cluster = ref.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# Fixed rendering colors per semantic team label (BGR)
TEAM_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (50, 50, 220),    # Team A — red
    1: (220, 100, 40),   # Team B — blue
    2: (150, 150, 150),  # Ref / Other — gray
}

TEAM_NAMES = {0: "Team A", 1: "Team B", 2: "Ref"}


class TeamClassifier:
    """
    Fits a 3-cluster K-means model on (H, S) torso color and assigns each
    tracked player a team label. Caches assignments by track ID so each
    player is consistently colored even when the torso crop is noisy.
    """

    def __init__(self, n_clusters: int = 3):
        self.k = n_clusters
        self._centers: Optional[np.ndarray] = None  # (k, 2) in HS space
        self._remap: Dict[int, int] = {}             # raw cluster idx → semantic label
        self._fitted: bool = False
        self._track_teams: Dict[int, int] = {}       # track_id → team label
        self._track_votes: Dict[int, List[int]] = {} # track_id → list of raw labels

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        frames: List[np.ndarray],
        all_detections: list,
        sample_every: int = 5,
    ) -> bool:
        """Fit K-means on torso color samples from a stride of frames."""
        features: List[np.ndarray] = []

        for i in range(0, len(frames), sample_every):
            frame = frames[i]
            for det in all_detections[i]:
                vec = self._hs_vec(frame, det)
                if vec is not None:
                    features.append(vec)

        if len(features) < self.k * 3:
            print(f"[team] Only {len(features)} samples — skipping fit (need ≥ {self.k * 3})")
            return False

        X = np.array(features, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(
            X, self.k, None, criteria, attempts=15,
            flags=cv2.KMEANS_PP_CENTERS,
        )

        self._centers = centers
        self._remap = self._build_remap(centers)
        self._fitted = True
        print(f"[team] K-means fitted on {len(features)} samples  "
              f"cluster centres (H,S): {centers.tolist()}")
        return True

    def classify(
        self,
        frame: np.ndarray,
        det,          # Detection
        track_id: int,
    ) -> int:
        """Return semantic team label (0/1/2). Caches per track_id via voting."""
        if not self._fitted:
            return 0

        vec = self._hs_vec(frame, det)
        if vec is None:
            return self._track_teams.get(track_id, 0)

        raw = int(np.argmin(np.linalg.norm(self._centers - vec, axis=1)))
        label = self._remap.get(raw, 0)

        # Accumulate votes and keep majority
        votes = self._track_votes.setdefault(track_id, [])
        votes.append(label)
        if len(votes) >= 5:
            # Lock after 5 observations
            from collections import Counter
            self._track_teams[track_id] = Counter(votes).most_common(1)[0][0]
        elif track_id not in self._track_teams:
            self._track_teams[track_id] = label

        return self._track_teams.get(track_id, label)

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        label = self._track_teams.get(track_id, 0)
        return TEAM_COLORS_BGR[label]

    def get_label(self, track_id: int) -> str:
        label = self._track_teams.get(track_id, 0)
        return TEAM_NAMES[label]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _hs_vec(self, frame: np.ndarray, det) -> Optional[np.ndarray]:
        """Extract mean (H, S) from torso region of a detection."""
        x1, y1 = int(det.x1), int(det.y1)
        x2, y2 = int(det.x2), int(det.y2)
        box_h = y2 - y1
        if box_h < 20 or (x2 - x1) < 10:
            return None

        # Torso: roughly 15–50 % from the top of the bounding box
        ty1 = y1 + int(box_h * 0.15)
        ty2 = y1 + int(box_h * 0.50)
        crop = frame[ty1:ty2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Exclude very dark pixels (shadows, arena floor)
        mask = hsv[:, :, 2] > 40
        if mask.sum() < 15:
            return None

        h_vals = hsv[:, :, 0][mask].astype(np.float32)
        s_vals = hsv[:, :, 1][mask].astype(np.float32)
        return np.array([h_vals.mean(), s_vals.mean()], dtype=np.float32)

    def _build_remap(self, centers: np.ndarray) -> Dict[int, int]:
        """
        Map raw cluster index → semantic label.
        Heuristic: lowest saturation (S channel) = referee.
        The two remaining clusters are assigned Team A / Team B by hue order.
        """
        saturations = centers[:, 1]  # S values
        order = np.argsort(saturations)  # ascending: [most-gray, ..., most-colorful]

        remap: Dict[int, int] = {}
        remap[int(order[0])] = 2  # least saturated → Ref
        remap[int(order[1])] = 0  # mid saturation  → Team A
        remap[int(order[2])] = 1  # most saturated  → Team B
        return remap
