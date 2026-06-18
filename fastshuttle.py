"""
fastshuttle.py -- catch a FAST shuttle by full-rate frame differencing
======================================================================

TrackNet (and slow frame-sampling) miss a hard-hit shuttle because at 60fps
it is a faint motion-blur streak present for only 1-3 frames. The classic fix
for small fast objects: look at EVERY consecutive frame and isolate what moved.

method per frame i:
    three-frame difference   min(|f_i - f_{i-1}|, |f_{i+1} - f_i|)  > thr
      -> keeps only genuinely MOVING pixels, kills static background AND the
         ghost trails a plain two-frame diff leaves behind
    connected components, keep small/compact blobs (a shuttle, not the arm)

then LINK per-frame blobs into short tracks (velocity-predicted nearest
neighbour). A struck shuttle is the track that travels FAR and FAST in a
fairly straight line -- that is what scan() scores and returns.

scan() takes a frame range so the full 16k-frame video can be swept in
parallel segments (see the workflow), each segment reading only its slice.
"""

import cv2
import numpy as np


def _blobs(prev, cur, nxt, thr, amin, amax, maxdim):
    d1 = np.abs(cur - prev)
    d2 = np.abs(nxt - cur)
    mov = (np.minimum(d1, d2) > thr).astype(np.uint8)
    mov = cv2.dilate(mov, np.ones((3, 3), np.uint8), iterations=1)
    n, lab, st, ct = cv2.connectedComponentsWithStats(mov, 8)
    out = []
    for k in range(1, n):
        a = int(st[k, cv2.CC_STAT_AREA])
        w = int(st[k, cv2.CC_STAT_WIDTH])
        h = int(st[k, cv2.CC_STAT_HEIGHT])
        if amin <= a <= amax and max(w, h) <= maxdim:
            out.append({"x": float(ct[k][0]), "y": float(ct[k][1]), "a": a})
    return out


def _link(blobs_by_frame, frame_ids, max_jump=260, max_gap=2, min_len=4,
          max_blobs=40):
    """Greedy velocity-predicted linking of per-frame blobs into tracks.

    Finished tracks (gap > max_gap) are retired immediately so the active set
    stays small -- without this a noisy segment accumulates thousands of dead
    tracks and linking blows up. Blobs per frame are capped (largest area
    first) to bound the work on compression-noise-heavy frames.
    """
    active, done = [], []
    for idx, fid in enumerate(frame_ids):
        bl = blobs_by_frame[idx]
        if len(bl) > max_blobs:
            bl = sorted(bl, key=lambda b: b["a"], reverse=True)[:max_blobs]
        used = [False] * len(bl)
        still = []
        for tr in active:
            lx, ly = tr["pts"][-1][1], tr["pts"][-1][2]
            px, py = lx + tr["vel"][0], ly + tr["vel"][1]   # predict
            best, bd = -1, max_jump
            for j, b in enumerate(bl):
                if used[j]:
                    continue
                d = np.hypot(b["x"] - px, b["y"] - py)
                if d < bd:
                    bd, best = d, j
            if best >= 0:
                b = bl[best]
                used[best] = True
                nvx, nvy = b["x"] - lx, b["y"] - ly
                tr["vel"] = (0.5 * tr["vel"][0] + 0.5 * nvx,
                             0.5 * tr["vel"][1] + 0.5 * nvy)
                tr["pts"].append((fid, b["x"], b["y"], b["a"]))
                tr["gap"] = 0
                still.append(tr)
            else:
                tr["gap"] += 1
                (still if tr["gap"] <= max_gap else done).append(tr)
        for j, b in enumerate(bl):
            if not used[j]:
                still.append({"pts": [(fid, b["x"], b["y"], b["a"])],
                              "vel": (0.0, 0.0), "gap": 0})
        active = still
    done.extend(active)
    return [t for t in done if len(t["pts"]) >= min_len]


def _score(tr, fps):
    pts = tr["pts"]
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    fids = np.array([p[0] for p in pts])
    seg = np.hypot(np.diff(xs), np.diff(ys))
    dt = np.diff(fids) / fps
    speeds = seg / np.where(dt == 0, 1e9, dt)
    disp = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    path = float(seg.sum())
    straight = disp / path if path > 0 else 0.0
    return {
        "f0": int(fids[0]), "f1": int(fids[-1]),
        "n": len(pts), "disp": disp, "path": path,
        "straightness": straight,
        "peak_speed": float(speeds.max()) if speeds.size else 0.0,
        "mean_area": float(np.mean([p[3] for p in pts])),
        "pts": [(int(p[0]), round(p[1], 1), round(p[2], 1)) for p in pts],
    }


def scan(video, start=0, end=None, thr=18, amin=8, amax=1800, maxdim=200,
         fps=None, min_disp=140, min_peak_speed=1500):
    """Scan [start, end) and return shuttle-like tracks (fast, far, smallish).

    Returns list of scored track dicts, sorted by a shuttle-likeness key.
    """
    from collections import deque
    cap = cv2.VideoCapture(video)
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = n_total if end is None else min(end, n_total)
    lo = max(0, start - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)

    # stream a 3-frame window so memory stays tiny (only blob centroids kept)
    buf = deque(maxlen=3)
    blobs_by_frame, frame_ids = [], []
    f = lo
    while f < end + 1:
        ok, fr = cap.read()
        if not ok:
            break
        buf.append((f, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.int16)))
        if len(buf) == 3:
            (_, g0), (f1, g1), (_, g2) = buf
            blobs_by_frame.append(_blobs(g0, g1, g2, thr, amin, amax, maxdim))
            frame_ids.append(f1)
        f += 1
    cap.release()
    if not frame_ids:
        return []

    tracks = _link(blobs_by_frame, frame_ids)
    scored = [_score(t, fps) for t in tracks]
    # shuttle-like: travels far AND fast (rejects jitter and the slow arm)
    shut = [s for s in scored if s["disp"] >= min_disp and s["peak_speed"] >= min_peak_speed]
    shut.sort(key=lambda s: s["disp"] * s["peak_speed"] * (0.5 + s["straightness"]),
              reverse=True)
    return shut


if __name__ == "__main__":
    # CLI: python fastshuttle.py <video> <start> <end> [amin]
    # prints compact JSON track summaries (no per-point lists) for the workflow
    import sys
    import json
    video = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    amin = int(sys.argv[4]) if len(sys.argv) > 4 else 18
    ts = scan(video, start=start, end=end, amin=amin)
    print(json.dumps([{
        "f0": t["f0"], "f1": t["f1"], "n": t["n"], "disp": round(t["disp"]),
        "peak": round(t["peak_speed"]), "area": round(t["mean_area"]),
        "straight": round(t["straightness"], 2),
        "x0": round(t["pts"][0][1]), "y0": round(t["pts"][0][2]),
        "x1": round(t["pts"][-1][1]), "y1": round(t["pts"][-1][2]),
    } for t in ts]))
