"""
Basketball Broadcast → Mini Replay Engine
==========================================
Usage:
    python main.py --input data/input.mp4 --config configs/default.yaml

    # Skip calibration UI (reuse saved calibration):
    python main.py --input data/input.mp4 --calibration outputs/calibration.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.video_io import load_frames, write_video
from src.calibration import run_calibration, load_calibration, COURT_W, COURT_H
from src.detection import PlayerDetector, BallDetector
from src.tracking import CentroidTracker
from src.projection import project_point
from src.render import draw_court, render_topdown_frame, render_overlay_frame
from src.team_classifier import TeamClassifier, TEAM_COLORS_BGR
from src.jersey_ocr import JerseyOCR


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Basketball Broadcast → Mini Replay Engine")
    p.add_argument("--input", default=None, help="Path to input .mp4 (overrides config)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--calibration", default=None,
                   help="Path to existing calibration JSON (skips interactive UI)")
    p.add_argument("--calib-frame", type=int, default=None,
                   help="Frame index to use for calibration")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    input_video = args.input or cfg["input_video"]
    video_stem = Path(input_video).stem          # e.g. "curry" from "data/curry.mp4"
    output_dir = Path(cfg["output_dir"]) / f"{video_stem}_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] Output directory: {output_dir}")

    video_cfg = cfg.get("video", {})
    det_cfg   = cfg.get("detection", {})
    trk_cfg   = cfg.get("tracking", {})
    proj_cfg  = cfg.get("projection", {})
    rnd_cfg   = cfg.get("render", {})
    team_cfg  = cfg.get("team_classifier", {})
    ball_cfg  = cfg.get("ball_detection", {})
    ocr_cfg   = cfg.get("jersey_ocr", {})

    # ── Step 1: Load video ────────────────────────────────────────────────────
    print("\n=== Step 1: Load Video ===")
    frames, fps = load_frames(
        input_video,
        resize_width=video_cfg.get("resize_width", 1280),
        every_nth=video_cfg.get("process_every_nth_frame", 1),
    )
    if not frames:
        sys.exit("[main] No frames loaded — check your input video path.")

    calib_frame_idx = args.calib_frame if args.calib_frame is not None \
        else cfg.get("calibration_frame_idx", 0)
    calib_frame_idx = min(calib_frame_idx, len(frames) - 1)
    calib_frame = frames[calib_frame_idx]
    cv2.imwrite(str(output_dir / "calibration_frame.jpg"), calib_frame)

    # ── Step 2: Calibration ───────────────────────────────────────────────────
    print("\n=== Step 2: Court Calibration ===")
    calib_path = output_dir / "calibration.json"
    calib_src = args.calibration or cfg.get("calibration_load")

    if calib_src and Path(calib_src).exists():
        H, calib_data = load_calibration(calib_src)
    else:
        print("[main] Opening interactive calibration window …")
        H, calib_data = run_calibration(
            calib_frame,
            preset=cfg.get("calibration_preset", "4corner"),
            save_path=str(calib_path),
        )

    court_w = calib_data.get("court_w", COURT_W)
    court_h = calib_data.get("court_h", COURT_H)
    warped = cv2.warpPerspective(calib_frame, H, (court_w, court_h))
    cv2.imwrite(str(output_dir / "calibration_debug.png"), warped)
    print("[main] Calibration debug image saved.")

    # ── Step 3: Player detection (pose) ───────────────────────────────────────
    print("\n=== Step 3: Player Detection (pose) ===")
    detector = PlayerDetector(
        confidence_threshold=det_cfg.get("confidence_threshold", 0.4),
    )
    all_detections = detector.detect_batch(frames)
    total_dets = sum(len(d) for d in all_detections)
    print(f"[main] Total player detections: {total_dets}")

    # ── Step 4: Ball detection ────────────────────────────────────────────────
    ball_enabled = ball_cfg.get("enabled", True)
    all_ball_positions = []
    if ball_enabled:
        print("\n=== Step 4: Ball Detection ===")
        ball_detector = BallDetector(
            confidence_threshold=ball_cfg.get("confidence_threshold", 0.25),
        )
        all_ball_positions = ball_detector.detect_batch(frames)
    else:
        all_ball_positions = [None] * len(frames)
        print("\n=== Step 4: Ball Detection (disabled) ===")

    # ── Step 5: Team color clustering ─────────────────────────────────────────
    print("\n=== Step 5: Team Color Clustering ===")
    team_clf = TeamClassifier(n_clusters=team_cfg.get("n_clusters", 3))
    team_fitted = team_clf.fit(
        frames,
        all_detections,
        sample_every=team_cfg.get("sample_every", 5),
    )
    if not team_fitted:
        print("[main] Team classifier not fitted — all players will use fallback colours.")

    # ── Step 6: Jersey OCR setup ──────────────────────────────────────────────
    ocr_enabled = ocr_cfg.get("enabled", True)
    jersey_ocr = JerseyOCR(
        read_every=ocr_cfg.get("read_every", 10),
        lock_votes=ocr_cfg.get("lock_votes", 3),
    ) if ocr_enabled else None
    if jersey_ocr:
        jersey_ocr.is_available()  # print availability message once upfront

    # ── Step 7: Tracking + Projection + Rendering ─────────────────────────────
    print("\n=== Step 7: Tracking + Projection + Rendering ===")
    tracker = CentroidTracker(
        max_match_distance=trk_cfg.get("max_match_distance", 60),
        max_stale_frames=trk_cfg.get("max_stale_frames", 8),
    )
    court_template = draw_court(court_w, court_h)
    clamp = proj_cfg.get("clamp_to_court", True)
    show_skeleton = rnd_cfg.get("show_skeleton", True)

    topdown_frames = []
    overlay_frames = []
    tracks_log = {}

    for i, (frame, dets) in enumerate(zip(frames, all_detections)):
        if i % 20 == 0:
            print(f"[main] Frame {i}/{len(frames)} ...")

        # Track
        matched = tracker.update(dets)

        # Build team colour map for this frame: {track_id: bgr_colour}
        frame_team_map: dict = {}
        for tid, det in matched.items():
            if team_fitted:
                label = team_clf.classify(frame, det, tid)
                frame_team_map[tid] = TEAM_COLORS_BGR[label]

        # Jersey OCR (per-track, throttled)
        frame_jersey_map: dict = {}
        if jersey_ocr:
            for tid, det in matched.items():
                jersey_ocr.read(frame, det, tid, i)
            # Collect all locked numbers for display
            for tid in matched:
                n = jersey_ocr.get_number(tid)
                if n is not None:
                    frame_jersey_map[tid] = n

        # Project player anchors → court space
        frame_court_positions: dict = {}
        for tid, det in matched.items():
            court_pos = project_point(det.anchor, H, court_w, court_h, clamp=clamp)
            tracker.tracks[tid].court_history.append(court_pos)
            if court_pos is not None:
                frame_court_positions[tid] = court_pos

        # Pad court_history for unmatched tracks
        for tid, t in tracker.tracks.items():
            if tid not in matched:
                t.court_history.append(None)

        # Project ball → court space
        ball_image = all_ball_positions[i] if i < len(all_ball_positions) else None
        ball_court: dict = None
        if ball_image is not None:
            ball_court = project_point(ball_image, H, court_w, court_h, clamp=clamp)

        # Render top-down
        td_frame = render_topdown_frame(
            court_template,
            frame_court_positions,
            tracker.tracks,
            frame_idx=i,
            team_map=frame_team_map if team_fitted else None,
            jersey_map=frame_jersey_map if frame_jersey_map else None,
            ball_court_pos=ball_court,
            trail_length=rnd_cfg.get("trail_length", 15),
            marker_radius=rnd_cfg.get("marker_radius", 8),
            show_ids=rnd_cfg.get("show_ids", True),
            show_trails=rnd_cfg.get("show_trails", True),
        )
        topdown_frames.append(td_frame)

        # Render overlay on original
        ov_frame = render_overlay_frame(
            frame,
            matched,
            team_map=frame_team_map if team_fitted else None,
            jersey_map=frame_jersey_map if frame_jersey_map else None,
            ball_image_pos=ball_image,
            show_ids=rnd_cfg.get("show_ids", True),
            show_skeleton=show_skeleton and rnd_cfg.get("show_skeleton", True),
        )
        overlay_frames.append(ov_frame)

        # Log
        tracks_log[i] = {
            str(tid): list(pos) for tid, pos in frame_court_positions.items()
        }
        if ball_court:
            tracks_log[i]["ball"] = list(ball_court)

    print(f"[main] Rendered {len(topdown_frames)} frames.")

    # ── Step 8: Save outputs ──────────────────────────────────────────────────
    print("\n=== Step 8: Saving Outputs ===")
    write_video(topdown_frames, str(output_dir / "output_topdown.mp4"), fps)
    write_video(overlay_frames, str(output_dir / "output_overlay.mp4"), fps)

    if topdown_frames:
        cv2.imwrite(str(output_dir / "preview_topdown.png"), topdown_frames[-1])

    with open(output_dir / "tracks.json", "w") as f:
        json.dump(tracks_log, f, indent=2)
    print(f"[main] tracks.json saved.")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n=== Done ===")
    print(f"  {output_dir}/output_topdown.mp4  — top-down replay (team colors, ball, jersey #s)")
    print(f"  {output_dir}/output_overlay.mp4  — annotated source (skeleton, team colors)")
    print(f"  {output_dir}/calibration.json    — homography")
    print(f"  {output_dir}/tracks.json         — frame-level court positions")


if __name__ == "__main__":
    main()
