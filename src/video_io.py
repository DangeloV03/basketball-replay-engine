import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


def load_frames(
    video_path: str,
    resize_width: int = 1280,
    every_nth: int = 1,
) -> Tuple[List[np.ndarray], float]:
    """Load frames from a video file.

    Returns:
        frames: list of BGR frames (optionally resized)
        fps: original video FPS
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video_io] Opening {path.name}: {total} frames @ {fps:.1f} FPS")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_nth == 0:
            if resize_width and frame.shape[1] != resize_width:
                scale = resize_width / frame.shape[1]
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (resize_width, new_h))
            frames.append(frame)

        frame_idx += 1

    cap.release()
    output_fps = fps / every_nth if every_nth > 1 else fps
    print(f"[video_io] Loaded {len(frames)} frames (every_nth={every_nth}, effective fps={output_fps:.1f})")
    return frames, output_fps


def write_video(frames: List[np.ndarray], output_path: str, fps: float) -> None:
    """Write a list of BGR frames to an mp4 file."""
    if not frames:
        print(f"[video_io] No frames to write to {output_path}")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(frame)

    writer.release()
    print(f"[video_io] Saved {len(frames)} frames → {output_path}")
