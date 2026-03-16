# Basketball Broadcast → Mini Replay Engine

A prototype computer vision system that takes a short basketball broadcast clip and reconstructs player movement as a top-down animated mini replay.

## What it does

1. Loads a local `.mp4` basketball clip
2. Lets you click 4–6 court landmarks on one frame to compute a homography
3. Detects players in every frame using **YOLOv8n** (no training required)
4. Tracks players across frames with a simple centroid tracker
5. Projects player positions into top-down court coordinates
6. Renders two output videos:
   - `output_topdown.mp4` — animated top-down replay with player dots and trails
   - `output_overlay.mp4` — original footage with bounding boxes and track IDs

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Step 1 — add your clip:**
```
data/input.mp4
```

**Step 2 — run:**
```bash
python main.py --input data/input.mp4 --config configs/default.yaml
```

A window will open showing a frame from the clip. Click the court landmarks **in order** as prompted (top-left → top-right → bottom-right → bottom-left for `4corner` preset). Press **`c`** to confirm, **`r`** to reset, **`q`** to quit.

**Reuse a saved calibration** (skips the click UI):
```bash
python main.py --input data/input.mp4 --calibration outputs/calibration.json
```

**Use a specific frame for calibration:**
```bash
python main.py --input data/input.mp4 --calib-frame 30
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/output_topdown.mp4` | Top-down replay animation |
| `outputs/output_overlay.mp4` | Original video with detection overlays |
| `outputs/calibration.json` | Homography matrix + reference points |
| `outputs/tracks.json` | Frame-by-frame court positions per track |
| `outputs/calibration_debug.png` | Warped calibration frame (sanity check) |
| `outputs/preview_topdown.png` | Last frame of the top-down replay |

## Configuration (`configs/default.yaml`)

Key knobs:

| Key | Default | Description |
|-----|---------|-------------|
| `calibration_preset` | `4corner` | `4corner` or `6point` (more accurate) |
| `calibration_frame_idx` | `0` | Which frame to use for calibration |
| `detection.confidence_threshold` | `0.4` | YOLO confidence cutoff |
| `tracking.max_match_distance` | `60` | Max pixel distance for centroid matching |
| `tracking.max_stale_frames` | `8` | Frames before a lost track is dropped |
| `render.show_trails` | `true` | Draw movement trails |
| `render.trail_length` | `15` | Number of past positions to trail |
| `video.process_every_nth_frame` | `1` | Subsample frames for speed |

## Tips

- **Best input:** 10–20 second half-court broadcast shot with clearly visible court lines.
- **Calibration accuracy matters most.** Use `6point` preset and pick spread-out landmarks.
- Check `outputs/calibration_debug.png` — if the warped court looks like a rectangle with court lines, calibration worked.
- If players appear off-court in the replay, recalibrate with more precise point placement.

## Stack

- Python 3.9+
- [OpenCV](https://opencv.org/) — video I/O, drawing, homography
- [NumPy](https://numpy.org/) — linear algebra
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — person detection (weights downloaded automatically)
- [PyYAML](https://pyyaml.org/) — config

## Limitations

- Manual court calibration required per clip (or per camera angle change)
- No ball detection
- No player identity — track IDs reset between runs
- Simple nearest-neighbour tracker can lose IDs when players cross paths
- Designed for steady broadcast/sideline cameras; wide fisheye or PTZ cameras will degrade homography quality

## Architecture

```
main.py              — pipeline orchestration
src/
  video_io.py        — load/write frames
  calibration.py     — interactive homography calibration
  detection.py       — YOLOv8n person detector
  tracking.py        — centroid tracker with nearest-neighbour matching
  projection.py      — homography-based point projection
  render.py          — court drawing + frame rendering
configs/
  default.yaml       — tunable parameters
data/                — input video
outputs/             — all generated artifacts
```
