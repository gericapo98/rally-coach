"""
extend.py -- motion-validated extension of each swing to its full in-frame arc
==============================================================================

On this footage the shuttle is OFF-FRAME between hits: it swings into the frame
during a strike, transits, and exits. So there is nothing to track between
hits -- the only honest continuity to gain is covering each swing's *full
in-frame transit*. `globaltrack.stitch` fits the detected core of a swing, but
the shuttle is usually visible (entering / exiting) for a few frames beyond
that core where the detector's far+fast filter didn't pick it up.

This module extends each fitted flight outward, frame by frame, toward the
frame edges. It does NOT use appearance correlation -- that traps on the
low-texture wall (it scores a blank wall as a confident match). Instead it uses
the one signal that is actually present where the shuttle is: MOTION. At each
extension frame it predicts the position from the flight's own trajectory
polynomial, looks for a real frame-difference blob near that prediction, and
snaps to it. It stops when the motion dies (the shuttle has left / stopped) or
the predicted path leaves the frame -- so the marker never wanders onto an
empty wall.

`extend_flights(video, flights, w, h, fps)` returns the same flight dicts with
longer, still-dense `path`/`speed` and updated `f0`/`f1`.
"""

import cv2
import numpy as np

import fastshuttle as fs


def _frame(cap, cache, f):
    """BGR frame `f`, memoized (extension re-reads neighbours for diff + colour)."""
    if f in cache:
        return cache[f]
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, im = cap.read()
    cache[f] = im if ok else None
    return cache[f]


def _blobs_at(cap, cache, f, thr=18, amin=8, amax=1800, maxdim=200):
    """Three-frame-difference motion blobs at frame `f` (reuses fastshuttle)."""
    f0, f1, f2 = _frame(cap, cache, f - 1), _frame(cap, cache, f), _frame(cap, cache, f + 1)
    if f0 is None or f1 is None or f2 is None:
        return []
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.int16)
    return fs._blobs(g(f0), g(f1), g(f2), thr, amin, amax, maxdim)


def _is_yellow(im, x, y, r=18, hue=(10, 40), sat_lo=70, val_lo=80, min_px=8):
    """True if a saturated-yellow (shuttle) blob sits at (x, y) in frame `im`."""
    if im is None:
        return False
    crop = im[max(0, int(y) - r):int(y) + r, max(0, int(x) - r):int(x) + r]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (hue[0], sat_lo, val_lo), (hue[1], 255, 255))
    return m.sum() / 255 >= min_px


def _extend_dir(cap, cache, flight, direction, w, h, max_ext, search, max_gap):
    """Walk a flight outward (+1 after f1, -1 before f0); return validated pts.

    Each step predicts the position from the flight polynomial and snaps to the
    nearest motion blob within `search` px THAT IS ALSO YELLOW (the shuttle's
    colour). Requiring yellow is what keeps the walk on the shuttle instead of
    drifting onto a compression-noise blob on the wall/floor/bottle. A few
    misses are tolerated (motion-blur), but `max_gap` consecutive misses -- or
    leaving the frame -- ends the walk.
    """
    px, py = flight["px"], flight["py"]
    anchor = flight["f1"] if direction > 0 else flight["f0"]
    pts, gap = [], 0
    for k in range(1, max_ext + 1):
        f = anchor + direction * k
        if f < 1:
            break
        cx, cy = float(np.polyval(px, f)), float(np.polyval(py, f))
        if not (0 <= cx < w and 0 <= cy < h):     # predicted to leave the frame
            break
        im = _frame(cap, cache, f)
        cand, bd = None, search
        for b in _blobs_at(cap, cache, f):
            d = np.hypot(b["x"] - cx, b["y"] - cy)
            if d < bd and _is_yellow(im, b["x"], b["y"]):
                bd, cand = d, b
        if cand is not None:
            pts.append((f, float(cand["x"]), float(cand["y"])))
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return pts


def _densify(knots):
    """Sorted (f, x, y) knots -> one (f, x, y) per integer frame via interp."""
    knots = sorted(knots)
    fs_ = [k[0] for k in knots]
    xs = [k[1] for k in knots]
    ys = [k[2] for k in knots]
    out = []
    for f in range(fs_[0], fs_[-1] + 1):
        out.append((f, float(np.interp(f, fs_, xs)), float(np.interp(f, fs_, ys))))
    return out


def color_support(cap, flight, r=26, hue=(10, 40), sat_lo=70, val_lo=80,
                  min_px=12, samples=12):
    """Fraction of a flight's sampled frames that show the YELLOW shuttle.

    The shuttle is a yellow nylon cock (H~10-40, S~130 in HSV) while the wall,
    lockers and floor it gets confused with are near-grey (low saturation). So
    a saturated-yellow blob at the marker is strong evidence the flight is on
    the real shuttle and not a compression-noise streak across the background.
    Returns a 0..1 confidence (sampled along the path).
    """
    p = flight["path"]
    step = max(1, len(p) // samples)
    hit = tot = 0
    for f, x, y in p[::step]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, im = cap.read()
        if not ok:
            continue
        xi, yi = int(x), int(y)
        crop = im[max(0, yi - r):yi + r, max(0, xi - r):xi + r]
        if crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (hue[0], sat_lo, val_lo), (hue[1], 255, 255))
        tot += 1
        hit += 1 if m.sum() / 255 >= min_px else 0
    return hit / tot if tot else 0.0


def gate_color(video, flights, min_support=0.18):
    """Keep only flights whose path rides the yellow shuttle (drops wall-noise).

    Conservative by design: the threshold is low enough to keep a genuine strike
    that looks white/desaturated under motion blur, while removing the clear
    cases where a smooth curve was fit through background compression noise.
    Annotates survivors with `color_support`.
    """
    cap = cv2.VideoCapture(video)
    kept = []
    for fl in flights:
        cs = color_support(cap, fl)
        if cs >= min_support:
            nf = dict(fl)
            nf["color_support"] = cs
            kept.append(nf)
    cap.release()
    return kept


def extend_flights(video, flights, w, h, fps, max_ext=24, search=80, max_gap=3):
    """Extend each flight to its full in-frame transit; returns updated flights."""
    if not flights:
        return flights
    cap = cv2.VideoCapture(video)
    out = []
    for fl in flights:
        cache = {}
        bwd = _extend_dir(cap, cache, fl, -1, w, h, max_ext, search, max_gap)
        fwd = _extend_dir(cap, cache, fl, +1, w, h, max_ext, search, max_gap)
        if not bwd and not fwd:
            out.append(fl)
            continue
        # knots: validated extension detections + the existing flight endpoints,
        # then re-densify so the path stays one point per frame.
        p = fl["path"]
        knots = bwd + [(p[0][0], p[0][1], p[0][2]), (p[-1][0], p[-1][1], p[-1][2])] + fwd
        ext = _densify(knots)
        # splice: extension head, original core (unchanged), extension tail
        head = [q for q in ext if q[0] < fl["f0"]]
        tail = [q for q in ext if q[0] > fl["f1"]]
        path = head + fl["path"] + tail

        f0, f1 = path[0][0], path[-1][0]
        speed = {f0: 0.0}
        for k in range(1, len(path)):
            (pf, pxv, pyv), (cf, cxv, cyv) = path[k - 1], path[k]
            dt = (cf - pf) / fps if cf != pf else 1e9
            speed[cf] = float(np.hypot(cxv - pxv, cyv - pyv) / dt)
        nf = dict(fl)
        nf.update(f0=f0, f1=f1, path=path, speed=speed,
                  peak_speed=max(speed.values()) if speed else fl["peak_speed"],
                  n_ext=len(head) + len(tail))
        out.append(nf)
    cap.release()
    return out
