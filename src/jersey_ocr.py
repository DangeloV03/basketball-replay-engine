"""
Jersey number OCR using PaddleOCR.

PaddleOCR is optional: if not installed the system runs without jersey numbers.
Install with:
    pip install paddlepaddle paddleocr

Reads jersey numbers by:
1. Cropping the chest/torso region of each bounding box.
2. Upscaling and enhancing contrast.
3. Running PaddleOCR and filtering for 1–2 digit results.
4. Caching per track ID with a voting strategy to reduce noise.
"""

import cv2
import numpy as np
from collections import Counter
from typing import Dict, List, Optional


class JerseyOCR:
    """
    Reads jersey numbers from detection crops.

    Runs lazily (every `read_every` frames per track) and locks a number
    in once `lock_votes` consistent reads are accumulated.
    """

    def __init__(self, read_every: int = 10, lock_votes: int = 3):
        self.read_every = read_every
        self.lock_votes = lock_votes
        self._reader = None
        self._available: Optional[bool] = None          # None = untested
        self._track_votes: Dict[int, List[int]] = {}    # track_id → list of candidate numbers
        self._track_numbers: Dict[int, int] = {}        # track_id → locked jersey number
        self._track_last_frame: Dict[int, int] = {}     # track_id → last frame we ran OCR

    # ── Public API ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from paddleocr import PaddleOCR  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                print("[jersey_ocr] PaddleOCR not found — jersey numbers disabled.\n"
                      "            Install with: pip install paddlepaddle paddleocr")
        return self._available

    def read(
        self,
        frame: np.ndarray,
        det,          # Detection
        track_id: int,
        frame_idx: int,
    ) -> Optional[int]:
        """
        Return the jersey number for this track (or None if unknown).
        OCR is only invoked when the number is not yet locked and
        enough frames have passed since the last attempt.
        """
        # Return cached locked number immediately
        if track_id in self._track_numbers:
            return self._track_numbers[track_id]

        # Throttle: only attempt OCR every `read_every` frames per track
        last = self._track_last_frame.get(track_id, -self.read_every)
        if frame_idx - last < self.read_every:
            return None

        if not self.is_available():
            return None

        self._track_last_frame[track_id] = frame_idx
        number = self._run_ocr(frame, det)

        if number is not None:
            votes = self._track_votes.setdefault(track_id, [])
            votes.append(number)
            if len(votes) >= self.lock_votes:
                winner = Counter(votes).most_common(1)[0][0]
                self._track_numbers[track_id] = winner
                print(f"[jersey_ocr] Track {track_id} → #{winner}  "
                      f"(from {votes})")
                return winner

        return None

    def get_number(self, track_id: int) -> Optional[int]:
        """Return cached number for a track (None if not yet determined)."""
        return self._track_numbers.get(track_id)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_reader(self):
        if self._reader is None:
            from paddleocr import PaddleOCR
            self._reader = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                show_log=False,
                use_gpu=False,
            )
        return self._reader

    def _run_ocr(self, frame: np.ndarray, det) -> Optional[int]:
        """Crop jersey region, preprocess, run OCR, return digit if found."""
        crop = self._jersey_crop(frame, det)
        if crop is None:
            return None

        enhanced = self._preprocess(crop)

        try:
            reader = self._get_reader()
            result = reader.ocr(enhanced, cls=False)
        except Exception as e:
            print(f"[jersey_ocr] OCR error: {e}")
            return None

        # Parse results: look for 1–2 digit strings with decent confidence
        for line in (result or []):
            for item in (line or []):
                try:
                    text, conf = item[1]
                except (TypeError, ValueError):
                    continue
                text = text.strip()
                if text.isdigit() and 1 <= len(text) <= 2 and conf >= 0.5:
                    number = int(text)
                    if 0 <= number <= 99:
                        return number
        return None

    def _jersey_crop(self, frame: np.ndarray, det) -> Optional[np.ndarray]:
        """
        Crop the jersey number region: roughly 20–60 % of bbox height,
        horizontally centered (trimmed slightly to exclude arms).
        """
        x1, y1 = int(det.x1), int(det.y1)
        x2, y2 = int(det.x2), int(det.y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 15 or bh < 30:
            return None

        # Vertical band: chest area
        cy1 = y1 + int(bh * 0.20)
        cy2 = y1 + int(bh * 0.60)
        # Horizontal: trim 15 % from each side to reduce arm confusion
        cx1 = x1 + int(bw * 0.15)
        cx2 = x2 - int(bw * 0.15)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        return crop

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Upscale + CLAHE contrast enhancement for better OCR accuracy."""
        h, w = crop.shape[:2]
        target = 96
        scale = max(1, target // min(h, w))
        if scale > 1:
            crop = cv2.resize(crop, (w * scale, h * scale),
                              interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
