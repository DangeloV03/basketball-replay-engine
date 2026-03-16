import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.detection import Detection


@dataclass
class Track:
    track_id: int
    # Image-space history
    history: List[Tuple[float, float]] = field(default_factory=list)       # centroids
    anchor_history: List[Tuple[float, float]] = field(default_factory=list)  # anchors
    # Court-space history (populated after projection)
    court_history: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    stale_frames: int = 0

    def update(self, det: Detection) -> None:
        self.history.append((det.cx, det.cy))
        self.anchor_history.append(det.anchor)
        self.stale_frames = 0

    def last_centroid(self) -> Tuple[float, float]:
        return self.history[-1] if self.history else (0.0, 0.0)


class CentroidTracker:
    """Simple nearest-neighbour centroid tracker."""

    def __init__(self, max_match_distance: float = 60.0, max_stale_frames: int = 8):
        self.max_dist = max_match_distance
        self.max_stale = max_stale_frames
        self._next_id = 0
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Detection]) -> Dict[int, Detection]:
        """Match detections to tracks. Returns {track_id: detection} for this frame."""
        # Purge stale tracks
        stale_ids = [tid for tid, t in self.tracks.items() if t.stale_frames > self.max_stale]
        for tid in stale_ids:
            del self.tracks[tid]

        active = list(self.tracks.values())
        matched: Dict[int, Detection] = {}
        unmatched_dets = list(detections)

        if active and unmatched_dets:
            track_pos = np.array([t.last_centroid() for t in active], dtype=np.float32)
            det_pos = np.array([(d.cx, d.cy) for d in unmatched_dets], dtype=np.float32)

            # Euclidean distance matrix (T × D)
            diff = track_pos[:, None, :] - det_pos[None, :, :]
            dist = np.linalg.norm(diff, axis=2)

            assigned_t: set = set()
            assigned_d: set = set()

            while True:
                if dist.size == 0:
                    break
                min_val = dist.min()
                if min_val > self.max_dist:
                    break
                t_idx, d_idx = np.unravel_index(dist.argmin(), dist.shape)
                if t_idx in assigned_t or d_idx in assigned_d:
                    dist[t_idx, d_idx] = np.inf
                    continue

                track = active[t_idx]
                det = unmatched_dets[d_idx]
                track.update(det)
                matched[track.track_id] = det
                assigned_t.add(t_idx)
                assigned_d.add(d_idx)
                dist[t_idx, :] = np.inf
                dist[:, d_idx] = np.inf

            unmatched_dets = [d for i, d in enumerate(unmatched_dets) if i not in assigned_d]

        # Mark unmatched existing tracks as stale
        matched_tids = set(matched.keys())
        for t in active:
            if t.track_id not in matched_tids:
                t.stale_frames += 1

        # Spawn new tracks for unmatched detections
        for det in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            t = Track(track_id=tid)
            t.update(det)
            self.tracks[tid] = t
            matched[tid] = det

        return matched
