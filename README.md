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

## Tracking the shuttle in the air (`globaltrack.py`)

The detector finds the shuttle when it *moves*, but a hard-hit shuttle is a
small fast object: at 60 fps it travels **400+ px/frame** and is a faint
motion-blur streak for only 1–3 frames at a time. Two detectors are wired up:

- **`shuttle.py`** — vendored **TrackNetV3** (appearance/heatmap). Trained on
  broadcast video; sparse on this phone footage.
- **`fastshuttle.py`** — full-rate three-frame differencing that finds every
  moving blob. High recall on the fast shuttle, but the footage also produces
  **hundreds of false fast blobs** from compression noise, and the greedy
  velocity-predicted linker **shatters one strike into many short tracklets**
  because the per-frame jump exceeds its gate and 2–4 frame blur gaps break it.

The original `render_hits.py` drew only the single best tracklet — on the
fastest hit that was **11 of ~460 frames (2%)**, so the marker flashed on for a
fraction of a second and the shuttle looked "lost."

### The approach: global trajectory stitching

`globaltrack.py` adds a tracking layer *on top of* the (untouched, known-good)
detector. Because there is only **one** shuttle, the multi-object association
machinery from the literature collapses to a cheap **seed-and-grow** scheme, run
**globally and offline** (an analysis tool can look forward *and* backward):

1. **Seed** on the strongest `scan` tracklets — long *and* fast *and* roughly
   straight. A tracklet like that is almost certainly a real strike, so it is a
   trustworthy trajectory hypothesis. Seeding on a coherent fast motion is what
   stops the fit from collapsing onto a static wobble (the failure mode of
   count-maximizing RANSAC on this footage).
2. **Robust-fit + grow** a smooth image-space motion model `x(f), y(f)`
   (quadratic in frame index, with trim-and-refit passes), absorbing every
   nearby detection on that curve. The fitted curve **bridges the 2–4 frame blur
   gaps by evaluation** (no invented detections) and rejects off-trajectory junk
   as outliers. Overlapping strikes in one burst each become their own clean
   flight; duplicates are deduped.

`render_hits.py` then draws **every** stitched flight with a marker on **every**
frame it covers, so the lock stays glued to the shuttle through each strike.
On a clean strike this turns the old 11-frame fragment into **~30 continuous
frames** tracking the blurred shuttle top-to-bottom.

### Why this design (grounded in a literature review)

A deep read of the SOTA (TrackNetV4 motion-attention, WASB, MonoTrack,
TrackNetV3, the min-cost-flow tracking lineage, and the motion-blur-as-velocity
work of Shishido et al.) pointed at three layers: a motion-aware detector, a
track-stitching layer, and a physics/trajectory prior. This module is the
**stitching + trajectory-prior** layer:

- the quadratic `x(f), y(f)` fit is the analysis-time, **image-space stand-in
  for MonoTrack's gravity+drag prior** — we can't use the real 3D physics model
  because the shuttle hangs on a **string** (a pendulum arc, not a free
  parabola) and there is **no court calibration** in this footage;
- gap-bridging-by-evaluation mirrors **TrackNetV3's trajectory-inpainting**
  idea without needing its learned module;
- seeding on far-and-fast tracklets is the offline cousin of **WASB's
  temporal-consistency gate**.

### Honest limitation (and the principled next step)

Because the candidate source is **motion-only**, this cannot fully reject
compression-noise streaks: noise can be long, straight, and fast enough to mimic
a strike, and simple appearance cues fail here (the white shuttle on a beige
wall has *lower* contrast than the background lockers; frame-diff energy at the
smoothed path is *higher* for some noise than for a real fast strike — both were
tested and rejected). So on the **clean single-strike hits the tracking is
tight**, while a **noisy multi-strike burst may still show an occasional marker
on the wall**. The principled fix, per the review, is an **appearance check
using the already-vendored TrackNet**: keep only fitted flights whose path the
heatmap also supports. That verification layer is the recommended next build.

## Layout

```
track_drill.py      racket-arm tracking + hit render    (drill)
analyze_drill.py    hits / swing speed / rhythm          (drill)
track_combined.py   full-body pose + shuttle render      (rally)
analyze.py          contacts / shot speed / heatmap      (rally)
players.py          YOLO11-pose wrapper (.infer / .arm)
shuttle.py          TrackNetV3 shuttle detector (appearance)
fastshuttle.py      full-rate frame-diff shuttle candidates (motion)
globaltrack.py      global trajectory stitching: tracklets -> flights
render_hits.py      slowed, shuttle-tracked clips of confirmed hits
localize.py         heatmap -> single shuttle point (MonoTrack §4.3)
TrackNetV3/         vendored pretrained shuttle model
```

The shuttle half (`shuttle.py`, `localize.py`, `TrackNetV3/`) is the proven
MonoTrack/TrackNetV2 recipe; the player, fast-detection, trajectory-stitching,
and coaching halves are new here.

## Limitations
- Speeds are in **pixels/second**. Add a 4-corner court/wall calibration to
  convert to km/h.
- Drill mode needs the wrist visible at contact; a hand fully off-frame at
  impact won't be measured.
- Rally-mode shuttle detection is sparse on phone footage (out-of-domain for
  the broadcast-trained model) — fine for trajectories, rough for exact speed.
- In-air shuttle tracking (`globaltrack.py`) is motion-only, so it can't fully
  reject compression-noise streaks on noisy phone footage — tight on clean
  strikes, occasionally a stray marker in a noisy multi-strike burst. The next
  step is an appearance check over each fitted flight using the vendored
  TrackNet (see "Tracking the shuttle in the air").
