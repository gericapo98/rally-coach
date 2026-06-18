"""
slowmo_speed.py -- slowed-down video with best-effort SHUTTLE speed tracking
============================================================================

Slow the video down and track the shuttle's speed as best as possible.

  SLOW MOTION   every source frame is kept and the output plays at
                fps/slowdown, so fast motion becomes easy to watch. No
                interpolation -> no fake frames.

  SHUTTLE SPEED two independent best-effort detectors run per frame:
                  (1) TrackNetV3  -- the broadcast-trained shuttle net
                  (2) motion blob -- online background subtraction, then the
                      small fast blob that ISN'T the big arm blob (--motion-blob)
                Speed is in PHYSICAL time (source fps), px/s, so the number is
                real regardless of the slow playback.

  RACKET SPEED  the racket-arm wrist (YOLO11-pose) is tracked too and its
                speed shown -- on plain-wall practice footage that is the only
                fast object that actually exists.

STREAMING: frames are processed one at a time (two passes over the file plus
TrackNet's own pass), so this runs on the whole 16k-frame video without
loading it all into memory.

    .venv/bin/python slowmo_speed.py 20260529_195739.mp4 --slowdown 2

Output: <name>_slowmo.mp4 (+ _slowmo.csv).  px/s only; add a court
calibration to convert to km/h.
"""

import os
import csv
import argparse
from collections import deque

import cv2
import numpy as np

import shuttle as shuttle_mod
from players import PlayerTracker

GREEN = (80, 220, 80)      # racket arm / wrist
YELLOW = (0, 255, 255)     # TrackNet shuttle
CYAN = (255, 220, 0)       # motion-blob shuttle candidate


def pick_blob(fg, arm_xy):
    """From a foreground mask, return the most shuttle-like blob centroid|None:
    small, compact, and well clear of the arm (so we don't relabel the hand)."""
    fg = cv2.medianBlur(fg, 5)
    n, lab, stats, cent = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), 8)
    ax, ay = arm_xy if arm_xy is not None else (None, None)
    best, best_score = None, -1.0
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 8 or area > 300:
            continue
        if max(stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]) > 60:
            continue
        cx, cy = cent[k]
        if ax is not None and np.hypot(cx - ax, cy - ay) < 300:
            continue
        score = 1.0 / (1 + abs(area - 60))
        if score > best_score:
            best_score, best = score, (float(cx), float(cy))
    return best


def speed_series(pts, fps):
    """px/s per frame from a list of (x,y)|None, gaps linearly bridged."""
    n = len(pts)
    xs = np.array([p[0] if p else np.nan for p in pts])
    ys = np.array([p[1] if p else np.nan for p in pts])
    good = ~np.isnan(xs)
    speed = np.full(n, np.nan)
    if good.sum() < 2:
        return speed
    idx = np.arange(n)
    xf = np.interp(idx, idx[good], xs[good])
    yf = np.interp(idx, idx[good], ys[good])
    v = np.hypot(np.gradient(xf), np.gradient(yf)) * fps
    speed[good] = v[good]
    return speed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("video")
    ap.add_argument("--slowdown", type=float, default=2.0,
                    help="playback slowdown factor (2 = 1/2 speed)")
    ap.add_argument("--thresh", type=float, default=0.3, help="TrackNet threshold")
    ap.add_argument("--box-conf", type=float, default=0.2)
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--motion-blob", action="store_true",
                    help="also draw the background-subtraction shuttle guess "
                         "(noisy on plain-wall footage; off by default)")
    ap.add_argument("--save-dir", default=os.path.join(os.path.dirname(__file__), "output"))
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"{name}_slowmo.mp4")
    csv_path = os.path.join(args.save_dir, f"{name}_slowmo.csv")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{n_total} frames {w}x{h} @ {fps:.1f}fps  -> playback {fps/args.slowdown:.1f}fps "
          f"(1/{args.slowdown:g}x, ~{n_total/(fps/args.slowdown)/60:.1f} min out)")

    # ---- pass 1 (stream): racket arm + online motion blob ----
    print("[pass 1/3] racket arm" + (" + motion blob" if args.motion_blob else "") + " ...")
    tracker = PlayerTracker(model=args.model)
    mog = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=20,
                                             detectShadows=False) if args.motion_blob else None
    wrist, mb_pts = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        a = tracker.arm(frame, box_conf=args.box_conf)
        wpt = a["wrist"] if a["present"] else None
        wrist.append(wpt)
        if mog is not None:
            fg = mog.apply(frame, learningRate=0.01)
            mb_pts.append(pick_blob(fg, wpt))
        else:
            mb_pts.append(None)
        i += 1
        if i % 1000 == 0:
            print(f"  [pass 1] {i}/{n_total}")
    cap.release()
    n = i

    # ---- TrackNet (its own streaming pass over the file) ----
    print("[pass 2/3] shuttle (TrackNetV3) ...")
    tn = shuttle_mod.detect_all(args.video, thresh=args.thresh)
    tn_pts = [tn[k]["uv"] if (k in tn and tn[k]["detected"]) else None for k in range(n)]

    wrist_speed = speed_series(wrist, fps)
    tn_speed = speed_series(tn_pts, fps)
    mb_speed = speed_series(mb_pts, fps)
    peak_wrist = np.nanmax(wrist_speed) if np.isfinite(wrist_speed).any() else 0.0
    print(f"  TrackNet shuttle pts: {sum(p is not None for p in tn_pts)}/{n}   "
          f"motion-blob pts: {sum(p is not None for p in mb_pts)}/{n}")
    print(f"  peak racket-arm speed: {peak_wrist:.0f} px/s")

    # ---- pass 3 (stream): draw + write slowed ----
    print("[pass 3/3] rendering slowed video ...")
    cap = cv2.VideoCapture(args.video)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                          max(1.0, fps / args.slowdown), (w, h))
    fh = open(csv_path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["frame", "time_s", "wrist_x", "wrist_y", "wrist_pxs",
                 "tracknet_x", "tracknet_y", "tracknet_pxs",
                 "motionblob_x", "motionblob_y", "motionblob_pxs"])
    trail = deque(maxlen=12)
    j = 0
    while True:
        ok, f = cap.read()
        if not ok or j >= n:
            break
        if wrist[j] is not None:
            wx, wy = int(wrist[j][0]), int(wrist[j][1])
            trail.append((wx, wy))
            cv2.circle(f, (wx, wy), 8, GREEN, -1)
            if np.isfinite(wrist_speed[j]):
                cv2.putText(f, f"{wrist_speed[j]:.0f} px/s", (wx + 12, wy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
        else:
            trail.clear()
        for k, (px, py) in enumerate(trail):
            fade = (k + 1) / len(trail)
            cv2.circle(f, (px, py), max(2, int(7 * fade)), GREEN, 1)
        if tn_pts[j] is not None:
            x, y = int(tn_pts[j][0]), int(tn_pts[j][1])
            cv2.circle(f, (x, y), 16, YELLOW, 3)
            cv2.putText(f, f"shuttle? {tn_speed[j]:.0f} px/s", (x + 18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
        if mb_pts[j] is not None:
            x, y = int(mb_pts[j][0]), int(mb_pts[j][1])
            cv2.drawMarker(f, (x, y), CYAN, cv2.MARKER_DIAMOND, 22, 2)

        cv2.rectangle(f, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.putText(f, f"t={j/fps:6.2f}s   1/{args.slowdown:g}x", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(f, f"racket peak {peak_wrist:.0f} px/s", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)
        cv2.putText(f, "yellow=TrackNet  cyan=motion  green=racket arm",
                    (w - 760, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        out.write(f)

        def s(p, c): return f"{p[c]:.1f}" if p else ""
        def sp(a): return f"{a[j]:.0f}" if np.isfinite(a[j]) else ""
        wr.writerow([j, f"{j/fps:.4f}", s(wrist[j], 0), s(wrist[j], 1), sp(wrist_speed),
                     s(tn_pts[j], 0), s(tn_pts[j], 1), sp(tn_speed),
                     s(mb_pts[j], 0), s(mb_pts[j], 1), sp(mb_speed)])
        j += 1
        if j % 1000 == 0:
            print(f"  [pass 3] {j}/{n}")
    cap.release()
    out.release()
    fh.close()
    print(f"\nslowmo video: {out_path}")
    print(f"csv:          {csv_path}")


if __name__ == "__main__":
    main()
