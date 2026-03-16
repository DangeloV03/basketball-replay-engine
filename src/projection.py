import cv2
import numpy as np
from typing import List, Optional, Tuple


def project_point(
    point: Tuple[float, float],
    H: np.ndarray,
    court_w: int,
    court_h: int,
    clamp: bool = True,
) -> Optional[Tuple[float, float]]:
    """Project a single image-space (x, y) to court coordinates via homography H."""
    src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)
    cx, cy = float(dst[0, 0, 0]), float(dst[0, 0, 1])

    if clamp:
        cx = max(0.0, min(float(court_w), cx))
        cy = max(0.0, min(float(court_h), cy))
    else:
        # Reject wildly out-of-bounds projections
        margin = max(court_w, court_h) * 0.5
        if not (-margin < cx < court_w + margin and -margin < cy < court_h + margin):
            return None

    return (cx, cy)


def project_points(
    points: List[Tuple[float, float]],
    H: np.ndarray,
    court_w: int,
    court_h: int,
    clamp: bool = True,
) -> List[Optional[Tuple[float, float]]]:
    """Project a list of image-space points to court coordinates."""
    return [project_point(p, H, court_w, court_h, clamp) for p in points]
