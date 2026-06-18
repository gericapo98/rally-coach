"""
render_hits.py -- slowed, shuttle-tracked clips of confirmed hits
=================================================================

Reads output/_shuttle_hits.json (written from the workflow's verified hits):
    [{"f0": int, "f1": int, "t0": float, "peak": float}, ...]

For each hit window it re-scans with fastshuttle, takes the best fast-object
track (the shuttle), and renders a SLOWED clip with the shuttle tracked:
    - the flight path drawn as it happens (orange polyline)
    - the shuttle marked each frame (yellow circle) + live px/s
    - header: time, "SHUTTLE", peak speed
Every source frame is kept; playback = fps/slowdown (no fake frames).

Outputs in output/:
    shuttle_hit_NN_t<seconds>.mp4   one per hit
    shuttle_hits_all.mp4            all hits concatenated

    .venv/bin/python render_hits.py <video> [--slowdown 4]
"""

import os
import json
import argparse
from collections import deque

import cv2
import numpy as np

import fastshuttle as fs

YELLOW = (0, 255, 255)
ORANGE = (0, 180, 255)


def track_speeds(top, fps):
    """px/s at each track point."""
    pts = top["pts"]
    sp = {pts[0][0]: 0.0}
    for k in range(1, len(pts)):
        f0, x0, y0 = pts[k - 1]
        f1, x1, y1 = pts[k]
        dt = (f1 - f0) / fps if f1 != f0 else 1e9
        sp[f1] = float(np.hypot(x1 - x0, y1 - y0) / dt)
    return sp


def render_window(cap, fps, w, h, f0, f1, top, slowdown, writer, label_idx):
    """Draw one hit window into `writer`. Returns frames written."""
    pts = {p[0]: (p[1], p[2]) for p in top["pts"]} if top else {}
    poly = [(p[0], int(p[1]), int(p[2])) for p in top["pts"]] if top else []
    speeds = track_speeds(top, fps) if top else {}
    peak = max(speeds.values()) if speeds else 0.0
    t0 = f0 / fps

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    written = 0
    for fid in range(f0, f1 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        # flight path up to this frame
        drawn = [p for p in poly if p[0] <= fid]
        for k in range(1, len(drawn)):
            cv2.line(fr, drawn[k - 1][1:], drawn[k][1:], ORANGE, 2)
        if fid in pts:
            x, y = int(pts[fid][0]), int(pts[fid][1])
            cv2.circle(fr, (x, y), 18, YELLOW, 3)
            cv2.putText(fr, f"{speeds.get(fid, 0):.0f} px/s", (x + 22, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2)
        cv2.rectangle(fr, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.putText(fr, f"SHUTTLE hit #{label_idx}   t={fid/fps:6.2f}s   1/{slowdown:g}x",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(fr, f"peak {peak:.0f} px/s", (20, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2)
        writer.write(fr)
        written += 1
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("video")
    ap.add_argument("--slowdown", type=float, default=4.0)
    ap.add_argument("--pad", type=float, default=0.4, help="seconds padded each side")
    ap.add_argument("--hits", default=None, help="hits json (default output/_shuttle_hits.json)")
    ap.add_argument("--save-dir", default=os.path.join(os.path.dirname(__file__), "output"))
    args = ap.parse_args()

    hits_path = args.hits or os.path.join(args.save_dir, "_shuttle_hits.json")
    hits = json.load(open(hits_path))
    if not hits:
        print("no shuttle hits to render")
        return

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = max(1.0, fps / args.slowdown)
    pad = int(args.pad * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    all_path = os.path.join(args.save_dir, "shuttle_hits_all.mp4")
    combined = cv2.VideoWriter(all_path, fourcc, out_fps, (w, h))

    made = []
    for i, hit in enumerate(sorted(hits, key=lambda d: d["f0"]), 1):
        f0 = max(0, int(hit["f0"]) - pad)
        f1 = min(n_total - 1, int(hit["f1"]) + pad)
        tracks = fs.scan(args.video, start=f0, end=f1 + 1)
        top = tracks[0] if tracks else None
        t0 = f0 / fps
        clip_path = os.path.join(args.save_dir, f"shuttle_hit_{i:02d}_t{t0:.1f}s.mp4")
        wclip = cv2.VideoWriter(clip_path, fourcc, out_fps, (w, h))
        n = render_window(cap, fps, w, h, f0, f1, top, args.slowdown, wclip, i)
        wclip.release()
        # also append into the combined reel
        render_window(cap, fps, w, h, f0, f1, top, args.slowdown, combined, i)
        peak = top["peak_speed"] if top else 0
        print(f"hit #{i}: t={t0:.1f}s  frames {f0}-{f1}  peak {peak:.0f} px/s  -> {clip_path}")
        made.append(clip_path)
    combined.release()
    cap.release()
    print(f"\ncombined reel: {all_path}")
    print(f"{len(made)} per-hit clips written")


if __name__ == "__main__":
    main()
