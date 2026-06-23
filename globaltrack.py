"""
globaltrack.py -- global single-object trajectory association for the shuttle
=============================================================================

The frame-difference detector (fastshuttle.scan) is a good *candidate* source
but a poor *tracker*: a hard-hit shuttle moves 400+ px/frame with 2-4 frame
motion-blur gaps, so the greedy velocity-predicted linker shatters one strike
into many short tracklets, and the renderer used to draw only the single best
one (~11 of ~460 frames -> the marker "loses" the shuttle).

This module does the association GLOBALLY and offline, which is the right
regime for an analysis tool (we can look forward AND backward). Because there
is only ONE shuttle, the multi-object min-cost-flow machinery from the
tracking literature collapses to a cheap, robust seed-and-grow scheme:

    1. SEED on the strongest tracklets. A `scan` tracklet that is long, fast,
       and reasonably straight is almost certainly a real strike; we sort these
       by disp x peak and use each as a trajectory hypothesis. Seeding on a
       coherent fast motion is what stops the fit from collapsing onto a static
       wobble -- the failure mode of count-maximizing RANSAC on this footage.

    2. ROBUST FIT + GROW. For each seed we robust-fit a smooth image-space
       motion model x(f), y(f) (quadratic in frame index, with trim-and-refit
       passes) and absorb every nearby detection that lies on that curve. This
         - resolves per-frame ambiguity (at 400 px/frame, position can't match
           across tracklets; a smooth model can),
         - bridges the 2-4 frame blur gaps by *evaluation* (no invented
           detections -- the curve is the interpolation),
         - rejects off-trajectory junk (e.g. a horizontal arm streak that
           overlaps the strike in time) as fit outliers.
    Seeds already claimed by a stronger flight are skipped, and a final dedupe
    drops residual duplicates.

    A quadratic-in-time prior is the analysis-time, image-space stand-in for
    MonoTrack's gravity+drag fit; we can't use the real 3D physics model here
    because the shuttle is on a string (a pendulum arc, not a free parabola)
    and there is no court calibration in this footage.

`stitch()` returns one dense, continuous trajectory per real strike, which
render_hits.py draws so the marker stays glued to the shuttle through each
strike. NOTE: because the candidate source is motion-only (frame differencing),
this cannot fully reject compression-noise streaks on noisy phone footage -- a
fundamental limitation discussed in the README, whose principled fix is an
appearance check (the vendored TrackNet) over the fitted path.
"""

import numpy as np


def _fit_xy(seed_pts, deg):
    """Fit polynomials x(f), y(f) to a list of (frame, x, y); clamp degree."""
    f = np.array([p[0] for p in seed_pts], dtype=np.float64)
    x = np.array([p[1] for p in seed_pts], dtype=np.float64)
    y = np.array([p[2] for p in seed_pts], dtype=np.float64)
    deg = min(deg, len(np.unique(f)) - 1)
    if deg < 1:
        return None
    return np.polyfit(f, x, deg), np.polyfit(f, y, deg), deg


def _robust_fit(pts, deg, tol, passes=2):
    """Least-squares x(f), y(f) with a couple of trim-and-refit passes.

    A `scan` tracklet can wander (its greedy linker pulls in off-position
    blobs), so a plain least-squares quadratic gets dragged off the true curve.
    Trimming the worst residuals over a few passes recovers the real trajectory
    before we use it to gather points. Returns (px, py, deg) or None.
    """
    cur = pts
    fit = _fit_xy(cur, deg)
    for _ in range(passes):
        if fit is None:
            return None
        px, py, d = fit
        f = np.array([p[0] for p in cur], dtype=np.float64)
        res = np.hypot(np.polyval(px, f) - [p[1] for p in cur],
                       np.polyval(py, f) - [p[2] for p in cur])
        keep = res <= tol * 1.5
        if keep.sum() < d + 1 or keep.all():
            break
        cur = [p for p, k in zip(cur, keep) if k]
        fit = _fit_xy(cur, deg)
    return fit


def _seed_flight(seed, pool, pool_f, pool_x, pool_y, tol, frame_pad, deg=2):
    """Fit a trajectory seeded on one tracklet, absorbing nearby pool points.

    Using a `scan` tracklet as the seed (instead of a random minimal sample) is
    what makes this robust: a tracklet is already one coherent fast motion, so
    its fit is the true strike curve -- it can't collapse onto a static blob the
    way count-maximizing RANSAC does. We robust-fit the seed, then gather every
    pool point that lies on that curve (residual < tol) within the seed's frame
    span +/- frame_pad, and refit on the gathered set. Returns (idx, px, py).
    """
    fit = _robust_fit(seed["pts"], deg, tol)
    if fit is None:
        return None
    px, py, d = fit
    f_lo, f_hi = seed["f0"] - frame_pad, seed["f1"] + frame_pad
    res = np.hypot(np.polyval(px, pool_f) - pool_x,
                   np.polyval(py, pool_f) - pool_y)
    idx = np.where((res <= tol) & (pool_f >= f_lo) & (pool_f <= f_hi))[0]
    if len(np.unique(pool_f[idx])) <= d:
        return None
    # one refit on the gathered consensus tightens the curve
    fit2 = _robust_fit([pool[i] for i in idx], deg, tol, passes=1)
    if fit2 is None:
        return None
    px, py, _ = fit2
    return idx, px, py


def _make_flight(f0, f1, n_inliers, px, py, fps, min_disp, max_span):
    """Build a dense per-frame flight dict from a fitted curve, or None."""
    if f1 - f0 + 1 > max_span:        # one quadratic spanning >1 strike -> reject
        return None
    path = [(f, float(np.polyval(px, f)), float(np.polyval(py, f)))
            for f in range(f0, f1 + 1)]
    if len(path) < 2:
        return None
    disp = float(np.hypot(path[-1][1] - path[0][1], path[-1][2] - path[0][2]))
    if disp < min_disp:
        return None
    speed = {f0: 0.0}
    for k in range(1, len(path)):
        (pf, pxv, pyv), (cf, cxv, cyv) = path[k - 1], path[k]
        dt = (cf - pf) / fps if cf != pf else 1e9
        speed[cf] = float(np.hypot(cxv - pxv, cyv - pyv) / dt)
    return {"f0": f0, "f1": f1, "path": path, "speed": speed, "disp": disp,
            "n_inliers": int(n_inliers),
            "peak_speed": max(speed.values()) if speed else 0.0,
            "px": np.asarray(px), "py": np.asarray(py)}  # for trajectory extension


def stitch(tracks, fps, tol=45, frame_pad=6, min_inliers=5, min_disp=400,
           deg=2, seed_disp=700, seed_peak=7000, seed_straight=0.5, seed_pts=5,
           pool_disp=350, pool_straight=0.5, max_span=60):
    """Tracklets -> one dense continuous trajectory per real strike.

    Pipeline: pre-filter tracklets to the genuinely shuttle-like ones, then use
    each surviving tracklet (strongest first) as a SEED for a smooth x(f), y(f)
    trajectory fit, absorbing every consistent detection around it. Seeds whose
    points are already mostly absorbed by a stronger flight are skipped, and a
    final dedupe drops residual duplicates. Seeding on coherent fast tracklets
    (rather than count-maximizing RANSAC) is what stops the fit from collapsing
    onto a static wobble and lets several overlapping strikes in one burst each
    become their own clean flight.

    Parameters
    ----------
    tracks      : list of fastshuttle.scan() track dicts (need f0,f1,pts,disp,
                  straightness,n)
    fps         : video fps (for per-frame px/s)
    tol         : px residual for a detection to lie on a trajectory
    frame_pad   : frames to extend a seed's span when gathering points
    min_inliers : detections required to accept a fitted flight
    min_disp    : reject fitted flights that don't travel this far (px)
    seed_disp, seed_peak, seed_straight, seed_pts : SEED filter (strict) -- only
                  a tracklet that travels far, moves fast, and is reasonably
                  straight may seed a flight. This is the precision lever: a
                  long+fast tracklet is almost certainly the real shuttle. Speed
                  (peak) matters as much as straightness, since a genuine strike
                  can wander a little while compression noise rarely travels far
                  AND fast together.
    pool_disp, pool_straight : POOL filter (loose) -- which tracklets' points a
                  seeded flight may absorb. Broader than the seed filter so a
                  real strike can pick up its own fragmented pieces, but still
                  tight enough that the robust fit + `tol` reject stray noise.
    max_span    : reject a flight whose frame span exceeds this (a single
                  quadratic spanning >1 strike)

    Returns
    -------
    list of flight dicts, sorted by supporting-detection count (strongest first):
        f0, f1    : first / last frame of the dense path
        path      : [(frame, x, y), ...] dense, one point per frame in [f0,f1]
        speed     : {frame: px/s} along the path
        disp      : straight-line start->end displacement (px)
        n_inliers : detections that supported the fit
        peak_speed: max px/s along the path
    """
    seeds = [t for t in tracks
             if t["disp"] >= seed_disp
             and t["peak_speed"] >= seed_peak
             and t["straightness"] >= seed_straight
             and t["n"] >= seed_pts]
    seeds.sort(key=lambda t: t["disp"] * t["peak_speed"], reverse=True)

    pool = [(int(f), float(x), float(y)) for t in tracks
            if t["disp"] >= pool_disp and t["straightness"] >= pool_straight
            for (f, x, y) in t["pts"]]
    if not seeds or not pool:
        return []
    pool_f = np.array([p[0] for p in pool], dtype=np.float64)
    pool_x = np.array([p[1] for p in pool], dtype=np.float64)
    pool_y = np.array([p[2] for p in pool], dtype=np.float64)

    flights = []
    claimed = set()                     # (frame, x, y) keys already in a flight
    for seed in seeds:
        # skip a seed whose own points are already mostly claimed by a flight
        sk = [(int(f), round(x, 1), round(y, 1)) for f, x, y in seed["pts"]]
        if sum(k in claimed for k in sk) > 0.5 * len(sk):
            continue
        fit = _seed_flight(seed, pool, pool_f, pool_x, pool_y, tol, frame_pad, deg)
        if fit is None:
            continue
        idx, px, py = fit
        if len(idx) < min_inliers:
            continue
        f0, f1 = int(pool_f[idx].min()), int(pool_f[idx].max())
        fl = _make_flight(f0, f1, len(idx), px, py, fps, min_disp, max_span)
        if fl is not None:
            flights.append(fl)
            for i in idx:
                claimed.add((pool[i][0], round(pool[i][1], 1), round(pool[i][2], 1)))

    flights.sort(key=lambda fl: fl["n_inliers"], reverse=True)
    return _dedupe(flights)


def _dedupe(flights, overlap_frac=0.5, pos_tol=150):
    """Drop flights that are near-duplicates of a stronger one.

    Sequential RANSAC can fit the same physical descent twice (slightly
    different quadratics over different point subsets). Two flights are dupes
    when their frame ranges overlap by more than `overlap_frac` of the shorter
    one AND their dense paths sit within `pos_tol` px over that overlap; the
    one with fewer supporting detections is discarded.
    """
    kept = []
    for fl in flights:                      # already sorted strongest-first
        pos = {f: (x, y) for f, x, y in fl["path"]}
        dup = False
        for kp in kept:
            kpos = {f: (x, y) for f, x, y in kp["path"]}
            shared = set(pos) & set(kpos)
            span = min(len(pos), len(kpos))
            if span and len(shared) >= overlap_frac * span:
                d = np.median([np.hypot(pos[f][0] - kpos[f][0],
                                        pos[f][1] - kpos[f][1]) for f in shared])
                if d <= pos_tol:
                    dup = True
                    break
        if not dup:
            kept.append(fl)
    return kept


if __name__ == "__main__":
    # diagnostic: print the stitched flights for a window of a video
    import sys
    import fastshuttle as fs_mod

    video = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    tracks = fs_mod.scan(video, start=start, end=end)
    flights = stitch(tracks, fps=60.0)
    print(f"{len(tracks)} raw tracklets -> {len(flights)} stitched flights")
    for i, fl in enumerate(flights):
        p = fl["path"]
        span = fl["f1"] - fl["f0"] + 1
        print(f"  flight {i}: frames {fl['f0']}-{fl['f1']} = {span}f continuous "
              f"({fl['n_inliers']} inliers)  y:{p[0][2]:.0f}->{p[-1][2]:.0f}  "
              f"disp={fl['disp']:.0f}  peak={fl['peak_speed']:.0f} px/s")
