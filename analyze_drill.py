"""
analyze_drill.py -- coaching numbers for the hanging-shuttle hitting drill
==========================================================================

This footage is a *drill*: a shuttle hangs on a string and you reach in to
strike it. There is no full-body player and no shuttle "flight" -- the only
reliable, useful signal is your RACKET ARM. track_drill.py extracts the
wrist (and elbow) per frame; this module turns that series into:

  1. HITS         a hit = an arm-present burst whose wrist speed peaks. The
                  contact instant is the peak-speed frame in that burst.
  2. SWING SPEED  peak + mean wrist pixel-speed (px/s) per hit. This is the
                  number to grow: a faster racket head = a harder shot.
  3. CONTACT HEIGHT  where the wrist is at contact, as a fraction of frame
                  height (1.0 = top of frame). Contacting high lets you hit
                  down; contacting low forces a lift.
  4. RHYTHM       inter-hit intervals, hits/min, and CONSISTENCY (lower
                  spread = more repeatable technique).

Shared helpers (arm_runs, wrist_speed, detect_hits) are imported by
track_drill.py so the video overlay and the report agree on the hits.

    .venv/bin/python analyze_drill.py output/<name>_drill.csv
"""

import os
import csv
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows)

    def col(name, typ=float):
        out = np.full(n, np.nan)
        for i, r in enumerate(rows):
            v = r.get(name, "")
            if v not in ("", None):
                out[i] = typ(v)
        return out

    t = col("time_s")
    present = np.array([r["arm_present"] == "1" for r in rows])
    return {"t": t, "wx": col("wrist_x"), "wy": col("wrist_y"),
            "score": col("wrist_score"), "present": present, "n": n}


def load_meta(csv_path):
    meta_path = csv_path.replace("_drill.csv", "_drill_meta.json")
    if os.path.exists(meta_path):
        return json.load(open(meta_path))
    return {"fps": 60.0, "width": 1080, "height": 1920}


def arm_runs(present, max_gap=4, min_len=2):
    """Group frame indices where the arm is in frame into bursts.
    Small gaps (<= max_gap frames of no detection) are bridged so a single
    swing that briefly drops a frame stays one burst.
    """
    runs, start, gap = [], None, 0
    for i, p in enumerate(present):
        if p:
            if start is None:
                start = i
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > max_gap:
                    if i - gap - start + 1 >= min_len:
                        runs.append((start, i - gap))
                    start = None
                    gap = 0
    if start is not None and len(present) - start >= min_len:
        runs.append((start, len(present) - 1))
    return runs


def wrist_speed(t, wx, wy):
    """Per-frame wrist speed (px/s), NaN where the wrist is unknown.
    Gaps inside a burst are linearly filled before differentiating."""
    n = len(t)
    speed = np.full(n, np.nan)
    good = ~np.isnan(wx)
    if good.sum() < 2:
        return speed
    idx = np.arange(n)
    xf = np.interp(idx, idx[good], wx[good])
    yf = np.interp(idx, idx[good], wy[good])
    dt = np.gradient(t)
    dt[dt == 0] = np.nan
    v = np.sqrt(np.gradient(xf) ** 2 + np.gradient(yf) ** 2) / dt
    speed[good] = v[good]
    return speed


def detect_hits(data, fps, min_peak_speed=900.0, max_gap=4, min_len=2):
    """Return a list of hits, one per qualifying arm burst.

    Each hit: {contact_idx, t, peak_speed, mean_speed, wrist (x,y),
               t0, t1, side?}  -- contact_idx is the peak-speed frame.
    """
    t, wx, wy = data["t"], data["wx"], data["wy"]
    speed = wrist_speed(t, wx, wy)
    hits = []
    for a, b in arm_runs(data["present"], max_gap=max_gap, min_len=min_len):
        seg = speed[a:b + 1]
        if not np.isfinite(seg).any():
            continue
        k = a + int(np.nanargmax(seg))
        peak = float(np.nanmax(seg))
        if peak < min_peak_speed:
            continue
        hits.append({
            "contact_idx": k, "t": float(t[k]), "peak_speed": peak,
            "mean_speed": float(np.nanmean(seg)),
            "wrist": (float(wx[k]), float(wy[k])),
            "t0": float(t[a]), "t1": float(t[b]),
        })
    return hits, speed


def speed_png(t, speed, hits, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, speed, "-", lw=0.8, color="C0", label="wrist speed (px/s)")
    for h in hits:
        ax.axvline(h["t"], color="C3", alpha=0.4)
        ax.plot(h["t"], h["peak_speed"], "o", color="C3", ms=5)
    ax.set_xlabel("time (s)"); ax.set_ylabel("wrist speed (px/s)")
    ax.set_title(f"Racket-arm swing speed, {len(hits)} hits")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def contact_png(hits, meta, path):
    W, H = meta["width"], meta["height"]
    fig, ax = plt.subplots(figsize=(5, 8))
    if hits:
        xs = [h["wrist"][0] for h in hits]
        ys = [h["wrist"][1] for h in hits]
        sp = [h["peak_speed"] for h in hits]
        sc = ax.scatter(xs, ys, c=sp, cmap="viridis", s=80, edgecolor="k")
        fig.colorbar(sc, ax=ax, label="peak swing speed (px/s)")
        for i, h in enumerate(hits):
            ax.annotate(str(i + 1), h["wrist"], fontsize=8,
                        textcoords="offset points", xytext=(5, 5))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title("Contact point of each hit\n(higher = contacted higher up)")
    ax.set_xlabel("x px"); ax.set_ylabel("y px")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def build_report(name, data, hits, meta):
    t = data["t"]
    H = meta["height"]
    dur = float(t[-1] - t[0]) if data["n"] else 0.0
    arm_frames = int(data["present"].sum())
    L, lines = (lambda s: lines.append(s)), []

    L(f"rally-coach :: hanging-shuttle drill :: {name}")
    L("=" * 52)
    L(f"duration        : {dur:.1f} s  ({data['n']} frames)")
    L(f"arm in frame    : {arm_frames}/{data['n']} frames "
      f"({100*arm_frames/max(1,data['n']):.0f}%)")
    L(f"hits detected   : {len(hits)}")
    if dur > 0:
        L(f"hitting tempo   : {len(hits)/dur*60:.1f} hits/min")
    L("")

    if hits:
        peaks = np.array([h["peak_speed"] for h in hits])
        heights = np.array([1 - h["wrist"][1] / H for h in hits])  # 1=top
        L("SWING SPEED (wrist px/s -- bigger = harder hit)")
        L(f"  fastest        : {peaks.max():6.0f} px/s   (hit #{int(peaks.argmax())+1})")
        L(f"  average        : {peaks.mean():6.0f} px/s")
        L(f"  slowest        : {peaks.min():6.0f} px/s   (hit #{int(peaks.argmin())+1})")
        cv = peaks.std() / peaks.mean() if peaks.mean() else 0
        L(f"  consistency    : {100*(1-cv):.0f}%  (spread {peaks.std():.0f} px/s; higher=more repeatable)")
        L("")
        L("CONTACT HEIGHT (fraction of frame; 1.0 = top)")
        L(f"  highest        : {heights.max():.2f}   lowest: {heights.min():.2f}   avg: {heights.mean():.2f}")
        L("")
        if len(hits) > 1:
            iv = np.diff([h["t"] for h in hits])
            L("RHYTHM (time between hits)")
            L(f"  interval avg   : {iv.mean():.2f} s   spread: {iv.std():.2f} s")
            L(f"  consistency    : {100*(1-iv.std()/iv.mean()) if iv.mean() else 0:.0f}%")
            L("")
        L("per hit:")
        L(f"  {'#':>2}  {'t(s)':>6}  {'peak px/s':>9}  {'mean px/s':>9}  {'height':>6}")
        for i, h in enumerate(hits):
            hh = 1 - h["wrist"][1] / H
            L(f"  {i+1:>2}  {h['t']:>6.2f}  {h['peak_speed']:>9.0f}  "
              f"{h['mean_speed']:>9.0f}  {hh:>6.2f}")
        L("")
        L("coaching read:")
        if cv > 0.35:
            L("  - swing speed varies a lot hit-to-hit: groove ONE repeatable")
            L("    motion before chasing more power.")
        else:
            L("  - swing speed is fairly repeatable: good base to add power on.")
        if heights.mean() < 0.55:
            L("  - you're contacting low on average: meet the shuttle EARLIER /")
            L("    higher to hit down instead of lifting.")
        else:
            L("  - nice high contact point -- keep meeting it out front and up.")
        L("  - px/s ranks effort already; add a 4-corner court calibration to")
        L("    convert to racket-head km/h.")
    else:
        L("No hits cleared the speed gate -- lower --min-peak-speed or check that")
        L("track_drill.py actually saw the arm (arm-in-frame % above).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("csv", help="<name>_drill.csv from track_drill.py")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--min-peak-speed", type=float, default=900.0,
                    help="min wrist peak speed (px/s) to count a burst as a hit")
    args = ap.parse_args()

    data = load(args.csv)
    meta = load_meta(args.csv)
    name = os.path.splitext(os.path.basename(args.csv))[0].replace("_drill", "")
    save_dir = args.save_dir or os.path.dirname(os.path.abspath(args.csv))

    hits, speed = detect_hits(data, meta["fps"], min_peak_speed=args.min_peak_speed)
    report = build_report(name, data, hits, meta)
    print(report)

    open(os.path.join(save_dir, f"{name}_drill_report.txt"), "w").write(report + "\n")
    speed_png(data["t"], speed, hits, os.path.join(save_dir, f"{name}_drill_speed.png"))
    contact_png(hits, meta, os.path.join(save_dir, f"{name}_drill_contact.png"))
    print(f"\nreport : {save_dir}/{name}_drill_report.txt")
    print(f"speed  : {save_dir}/{name}_drill_speed.png")
    print(f"contact: {save_dir}/{name}_drill_contact.png")


if __name__ == "__main__":
    main()
