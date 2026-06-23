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

import cv2

import fastshuttle as fs
import globaltrack as gt
import extend as ext

YELLOW = (0, 255, 255)
ORANGE = (0, 180, 255)


def _frame_index(flights):
    """Build {frame -> (x, y, speed, flight_id)} from all stitched flights.

    Where flights overlap on a frame, the marker for the one with more
    supporting detections wins, so the drawn point stays on the dominant
    shuttle motion.
    """
    idx = {}
    order = sorted(range(len(flights)), key=lambda i: flights[i]["n_inliers"])
    for fi in order:                         # weakest first so strongest overwrites
        fl = flights[fi]
        for f, x, y in fl["path"]:
            idx[f] = (x, y, fl["speed"].get(f, 0.0), fi)
    return idx


def render_window(cap, fps, w, h, f0, f1, flights, slowdown, writer, label_idx):
    """Draw one hit window into `writer`. Returns frames written.

    Every stitched flight is drawn as a growing orange polyline, with a yellow
    marker on the shuttle at every frame the flight covers (gaps filled by the
    fitted trajectory), so the marker stays glued to the shuttle through each
    strike in the burst instead of flashing on for one fragment.
    """
    fidx = _frame_index(flights)
    polys = [[(int(p[0]), int(p[1]), int(p[2])) for p in fl["path"]]
             for fl in flights]
    peak = max((fl["peak_speed"] for fl in flights), default=0.0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    written = 0
    for fid in range(f0, f1 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        # per-frame appearance gate: only mark a frame where the predicted spot
        # is actually the saturated yellow shuttle (checked on the clean frame,
        # before any overlay). This suppresses the frames where a flight's path
        # drifts off the shuttle onto the wall/floor between detections.
        show = fid in fidx and ext._is_yellow(fr, fidx[fid][0], fidx[fid][1])
        # each flight's path up to this frame
        for poly in polys:
            drawn = [p for p in poly if p[0] <= fid]
            for k in range(1, len(drawn)):
                cv2.line(fr, drawn[k - 1][1:], drawn[k][1:], ORANGE, 2)
        if show:
            x, y, spd, _ = fidx[fid]
            x, y = int(x), int(y)
            cv2.circle(fr, (x, y), 18, YELLOW, 3)
            cv2.putText(fr, f"{spd:.0f} px/s", (x + 22, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2)
        cv2.rectangle(fr, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.putText(fr, f"SHUTTLE hit #{label_idx}   t={fid/fps:6.2f}s   1/{slowdown:g}x",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(fr, f"peak {peak:.0f} px/s   {len(flights)} flights",
                    (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2)
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
        flights = gt.stitch(tracks, fps)
        flights = ext.gate_color(args.video, flights)        # drop wall-noise (yellow check)
        flights = ext.extend_flights(args.video, flights, w, h, fps)  # full in-frame swing
        t0 = f0 / fps
        clip_path = os.path.join(args.save_dir, f"shuttle_hit_{i:02d}_t{t0:.1f}s.mp4")
        wclip = cv2.VideoWriter(clip_path, fourcc, out_fps, (w, h))
        n = render_window(cap, fps, w, h, f0, f1, flights, args.slowdown, wclip, i)
        wclip.release()
        # also append into the combined reel
        render_window(cap, fps, w, h, f0, f1, flights, args.slowdown, combined, i)
        peak = max((fl["peak_speed"] for fl in flights), default=0)
        print(f"hit #{i}: t={t0:.1f}s  frames {f0}-{f1}  {len(flights)} flights  "
              f"peak {peak:.0f} px/s  -> {clip_path}")
        made.append(clip_path)
    combined.release()
    cap.release()
    print(f"\ncombined reel: {all_path}")
    print(f"{len(made)} per-hit clips written")


if __name__ == "__main__":
    main()
