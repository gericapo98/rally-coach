"""
analyze.py -- turn the track CSV into coaching numbers
======================================================

Reads <name>_track.csv (from track_combined.py) and produces signals that
actually help you hit better:

  1. CONTACT / HIT MOMENTS
     A hit is where the shuttle's vertical direction reverses sharply -- the
     cork goes up after a clear, or the trajectory kinks at a racket strike.
     We smooth the shuttle path, find sign changes in the vertical velocity
     with enough prominence, and call each one a contact. For each contact we
     attribute it to whichever player's wrist is closest.

  2. SHUTTLE SPEED PER SHOT
     Between consecutive contacts the shuttle flies roughly ballistically.
     We report peak and mean pixel-speed (px/s) over each flight; with a court
     calibration these become km/h, but px/s already ranks your shots.

  3. PLAYER COURT-POSITION HEATMAP
     Where each player's feet spend time. Standing too deep / stuck on one
     side shows up immediately.

Outputs to --save-dir (default ./output):
    <name>_report.txt       human-readable summary
    <name>_heatmap.png      per-player foot-position heatmap
    <name>_shuttle.png      shuttle height vs time with contacts marked

    .venv/bin/python analyze.py output/clip_track.csv
"""

import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows)
    t = np.array([float(r["time_s"]) for r in rows])

    def col(name):
        out = np.full(n, np.nan)
        for i, r in enumerate(rows):
            v = r.get(name, "")
            if v not in ("", None):
                out[i] = float(v)
        return out

    data = {
        "t": t,
        "sx": col("sx"), "sy": col("sy"),
        "p1_foot": (col("p1_foot_x"), col("p1_foot_y")),
        "p2_foot": (col("p2_foot_x"), col("p2_foot_y")),
        "p1_wrists": (col("p1_lwrist_x"), col("p1_lwrist_y"),
                      col("p1_rwrist_x"), col("p1_rwrist_y")),
        "p2_wrists": (col("p2_lwrist_x"), col("p2_lwrist_y"),
                      col("p2_rwrist_x"), col("p2_rwrist_y")),
    }
    return data, rows


def interp_nan(y):
    """Linear-fill NaN gaps so we can differentiate the trajectory."""
    y = y.copy()
    idx = np.arange(len(y))
    good = ~np.isnan(y)
    if good.sum() < 2:
        return y
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


def find_contacts(t, sy, smooth_sigma=2.0, min_gap_s=0.25, min_prominence=6.0):
    """Contacts = prominent reversals in vertical shuttle motion.

    Image y grows downward, so a local MIN of y is the top of an arc and a
    local change of vertical-velocity sign is a strike or bounce. We detect
    sign changes in the smoothed vertical velocity, keep those with enough
    local amplitude, and de-duplicate within min_gap_s.
    """
    if np.isfinite(sy).sum() < 5:
        return []
    ys = gaussian_filter1d(interp_nan(sy), smooth_sigma)
    vy = np.gradient(ys)
    sign = np.sign(vy)
    cand = np.where(np.diff(sign) != 0)[0] + 1

    contacts = []
    last_t = -1e9
    half = 5  # frames around the reversal to measure amplitude
    for i in cand:
        lo, hi = max(0, i - half), min(len(ys), i + half)
        amp = ys[lo:hi].max() - ys[lo:hi].min()
        if amp < min_prominence:
            continue
        if t[i] - last_t < min_gap_s:
            continue
        contacts.append(i)
        last_t = t[i]
    return contacts


def attribute(data, idx):
    """Which player struck this contact? Nearest wrist to the shuttle."""
    sx, sy = data["sx"][idx], data["sy"][idx]
    if np.isnan(sx):
        return None
    best, best_d = None, 1e18
    for n in (1, 2):
        lwx, lwy, rwx, rwy = data[f"p{n}_wrists"]
        for wx, wy in ((lwx[idx], lwy[idx]), (rwx[idx], rwy[idx])):
            if np.isnan(wx):
                continue
            d = (wx - sx) ** 2 + (wy - sy) ** 2
            if d < best_d:
                best_d, best = d, n
    return best


def shot_speeds(t, sx, sy, contacts):
    """Peak/mean shuttle pixel-speed over each inter-contact flight."""
    sxx, syy = interp_nan(sx), interp_nan(sy)
    speed = np.full(len(t), np.nan)
    dt = np.gradient(t)
    dt[dt == 0] = np.nan
    v = np.sqrt(np.gradient(sxx) ** 2 + np.gradient(syy) ** 2) / dt
    speed[:] = v
    shots = []
    bounds = contacts + [len(t) - 1]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = speed[a + 1:b]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            shots.append({"t0": float(t[a]), "t1": float(t[b]),
                          "peak": float(seg.max()), "mean": float(seg.mean())})
    return shots


def heatmap_png(data, t, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, n in zip(axes, (1, 2)):
        fx, fy = data[f"p{n}_foot"]
        good = ~np.isnan(fx)
        ax.set_title(f"Player {n} court position  ({good.sum()} frames)")
        if good.sum() > 3:
            H, xe, ye = np.histogram2d(fx[good], fy[good], bins=40)
            H = gaussian_filter(H, 1.5)
            ax.imshow(H.T, origin="upper",
                      extent=[xe[0], xe[-1], ye[-1], ye[0]],
                      aspect="auto", cmap="hot")
            ax.scatter(np.nanmean(fx), np.nanmean(fy), c="cyan", marker="+",
                       s=120, label="mean")
            ax.legend(loc="upper right")
        ax.set_xlabel("x px"); ax.set_ylabel("y px")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def shuttle_png(t, sy, contacts, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, sy, ".", ms=2, color="0.6", label="shuttle y (raw)")
    ax.plot(t, gaussian_filter1d(interp_nan(sy), 2.0), "-", lw=1, color="C0",
            label="smoothed")
    for i in contacts:
        ax.axvline(t[i], color="C3", alpha=0.5)
    ax.invert_yaxis()  # image y grows downward; flip so "up" is up
    ax.set_xlabel("time (s)"); ax.set_ylabel("shuttle height (px, flipped)")
    ax.set_title(f"Shuttle trajectory, {len(contacts)} contacts")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("csv", help="<name>_track.csv from track_combined.py")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--min-prominence", type=float, default=6.0,
                    help="min vertical amplitude (px) for a contact")
    args = ap.parse_args()

    data, rows = load(args.csv)
    t = data["t"]
    name = os.path.splitext(os.path.basename(args.csv))[0].replace("_track", "")
    save_dir = args.save_dir or os.path.dirname(os.path.abspath(args.csv))

    contacts = find_contacts(t, data["sy"], min_prominence=args.min_prominence)
    shots = shot_speeds(t, data["sx"], data["sy"], contacts)

    # build report
    lines = []
    L = lines.append
    L(f"rally-coach report :: {name}")
    L("=" * 50)
    dur = t[-1] - t[0] if len(t) else 0
    sh_det = np.isfinite(data["sx"]).sum()
    L(f"duration       : {dur:.1f} s  ({len(t)} frames)")
    L(f"shuttle seen   : {sh_det}/{len(t)} frames ({100*sh_det/max(1,len(t)):.0f}%)")
    L(f"contacts/hits  : {len(contacts)}")
    if dur > 0:
        L(f"rally tempo    : {len(contacts)/dur*60:.1f} hits/min")
    L("")
    L("per-contact (who hit, by nearest wrist):")
    by_player = {1: 0, 2: 0, None: 0}
    for i in contacts:
        who = attribute(data, i)
        by_player[who] = by_player.get(who, 0) + 1
        L(f"  t={t[i]:6.2f}s  player={who if who else '?'}")
    L(f"  -> player1: {by_player.get(1,0)}   player2: {by_player.get(2,0)}"
      f"   unknown: {by_player.get(None,0)}")
    L("")
    L("shot speeds (shuttle, px/s over each flight):")
    if shots:
        peaks = [s["peak"] for s in shots]
        L(f"  shots measured : {len(shots)}")
        L(f"  fastest peak   : {max(peaks):.0f} px/s  (t={shots[int(np.argmax(peaks))]['t0']:.2f}s)")
        L(f"  median peak    : {np.median(peaks):.0f} px/s")
        for s in shots:
            L(f"  t={s['t0']:6.2f}->{s['t1']:6.2f}s  peak={s['peak']:6.0f}  mean={s['mean']:6.0f} px/s")
    else:
        L("  (no inter-contact flights measured)")
    L("")
    L("tip: px/s ranks your shots already; add a court calibration")
    L("     (4 corner points) to convert to km/h.")

    report = "\n".join(lines)
    print(report)
    rep_path = os.path.join(save_dir, f"{name}_report.txt")
    open(rep_path, "w").write(report + "\n")

    hm_path = os.path.join(save_dir, f"{name}_heatmap.png")
    sh_path = os.path.join(save_dir, f"{name}_shuttle.png")
    heatmap_png(data, t, hm_path)
    shuttle_png(t, data["sy"], contacts, sh_path)

    print(f"\nreport : {rep_path}")
    print(f"heatmap: {hm_path}")
    print(f"shuttle: {sh_path}")


if __name__ == "__main__":
    main()
