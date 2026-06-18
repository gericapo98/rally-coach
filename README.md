# rally-coach

Track yourself + the shuttle from badminton practice video and turn it into
numbers you can train against. Two modes, because real footage comes in two
very different shapes:

| mode | use when | tracks | run |
|------|----------|--------|-----|
| **drill** | wall/hanging-shuttle practice — you reach in to hit a shuttle on a string, no full body in frame | your racket-arm **wrist** + every **hit** | `track_drill.py` → `analyze_drill.py` |
| **rally** | normal play — your whole body is in frame and the shuttle flies | full-body **pose** + **shuttle flight** + court heatmap | `track_combined.py` → `analyze.py` |

The video that seeded this project (`20260529_195739.mp4`) is a **drill**: a
shuttle hangs on a string against a wall and only your forearm enters the
frame at each strike. So `drill` mode is the one that produces real numbers
on it; `rally` mode is built and validated but needs footage where you're
actually on court.

## Setup

```bash
uv venv --python 3.11
uv pip install ultralytics opencv-python numpy scipy matplotlib parse pandas tqdm lap
```

YOLO11-pose weights (`yolo11m-pose.pt`) auto-download on first run. The
shuttle detector (rally mode) uses the vendored **TrackNetV3** checkpoints in
`TrackNetV3/ckpts/` (`*.pt`, gitignored — copy them in if missing).

## Drill mode (this footage)

```bash
# 1. track the racket arm + detect hits -> annotated video + csv
.venv/bin/python track_drill.py 20260529_195739.mp4

# 2. coaching report + plots
.venv/bin/python analyze_drill.py output/20260529_195739_drill.csv
```

Outputs in `output/`:
- `*_drill.mp4` — wrist dot + forearm + fading trail, live **px/s** readout,
  a red **HIT** flash at each contact
- `*_drill_report.txt` — hits, **swing speed** (peak/avg/consistency),
  **contact height**, **rhythm**
- `*_drill_speed.png` — wrist speed over time, hits marked
- `*_drill_contact.png` — where each hit was contacted, colored by speed

**What to train on:** grow *peak px/s* (racket-head speed), raise *contact
height*, and tighten *consistency* (repeat the same swing).

### knobs
- `--min-peak-speed` (default 900) — raise to ignore soft taps, lower if real
  hits are missed
- `--box-conf` (default 0.2) — YOLO confidence for the partial-arm detection

## Rally mode (full-body footage)

```bash
.venv/bin/python track_combined.py output/clip.mp4   # players + shuttle
.venv/bin/python analyze.py output/clip_track.csv    # contacts, speeds, heatmap
```

`--thresh` defaults to **0.3**: the TrackNetV3 paper's 0.5 finds nothing on
phone footage (it's trained on landscape broadcast), 0.3 recovers the shuttle.

## Layout

```
track_drill.py      racket-arm tracking + hit render   (drill)
analyze_drill.py    hits / swing speed / rhythm         (drill)
track_combined.py   full-body pose + shuttle render     (rally)
analyze.py          contacts / shot speed / heatmap     (rally)
players.py          YOLO11-pose wrapper (.infer / .arm)
shuttle.py          TrackNetV3 shuttle detector
localize.py         heatmap -> single shuttle point (MonoTrack §4.3)
TrackNetV3/         vendored pretrained shuttle model
```

The shuttle half (`shuttle.py`, `localize.py`, `TrackNetV3/`) is the proven
MonoTrack/TrackNetV2 recipe; the player and coaching halves are new here.

## Limitations
- Speeds are in **pixels/second**. Add a 4-corner court/wall calibration to
  convert to km/h.
- Drill mode needs the wrist visible at contact; a hand fully off-frame at
  impact won't be measured.
- Rally-mode shuttle detection is sparse on phone footage (out-of-domain for
  the broadcast-trained model) — fine for trajectories, rough for exact speed.
