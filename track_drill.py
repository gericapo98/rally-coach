"""
track_drill.py -- track the racket arm in the hanging-shuttle drill
===================================================================

Tailored to THIS footage: a shuttle hangs on a string and you reach in to
hit it. No full body is ever in frame, so we don't fake a skeleton or a
shuttle flight. We track the one thing that's real and useful -- your
racket-arm WRIST -- and mark every hit.

Two passes:
    pass 1  YOLO11-pose per frame -> wrist + elbow series (PlayerTracker.arm)
    pass 2  detect hits from the series, then render:
              - forearm segment (elbow->wrist) + wrist dot + fading trail
              - live wrist speed (px/s) readout
              - a "HIT  <peak> px/s" flash around each contact frame

Outputs (in --save-dir, default ./output):
    <name>_drill.mp4         annotated video
    <name>_drill.csv         frame, time_s, arm_present, wrist_x/y, wrist_score
    <name>_drill_meta.json   fps/width/height (analyze_drill.py reads it)

    .venv/bin/python track_drill.py output/clip.mp4
then:
    .venv/bin/python analyze_drill.py output/clip_drill.csv
"""

import os
import csv
import json
import argparse
from collections import deque

import cv2
import numpy as np

from players import PlayerTracker
from analyze_drill import detect_hits, wrist_speed

ARM_COLOR = (80, 200, 80)
TRAIL_COLOR = (0, 255, 255)
HIT_COLOR = (60, 60, 255)


def collect(video_file, tracker, box_conf):
    """Pass 1: per-frame wrist/elbow series."""
    cap = cv2.VideoCapture(video_file)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    series = {"wx": [], "wy": [], "score": [], "present": [], "elbow": []}
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        a = tracker.arm(frame, box_conf=box_conf)
        if a["present"]:
            series["wx"].append(a["wrist"][0]); series["wy"].append(a["wrist"][1])
            series["score"].append(a["score"]); series["present"].append(True)
            series["elbow"].append(a["elbow"])
        else:
            series["wx"].append(np.nan); series["wy"].append(np.nan)
            series["score"].append(np.nan); series["present"].append(False)
            series["elbow"].append(None)
        f += 1
        if f % 600 == 0:
            print(f"  [pass 1] {f}/{n} frames")
    cap.release()
    for k in ("wx", "wy", "score"):
        series[k] = np.array(series[k], dtype=float)
    series["present"] = np.array(series["present"], dtype=bool)
    return series


def render(video_file, series, hits, fps, csv_path, video_path, meta_path, trail=15):
    t = np.arange(len(series["present"])) / fps
    data = {"t": t, "wx": series["wx"], "wy": series["wy"],
            "present": series["present"], "n": len(t)}
    speed = wrist_speed(t, series["wx"], series["wy"])

    # frames near a contact get a HIT flash (+/- ~0.15s)
    flash = {}
    half = max(1, int(0.15 * fps))
    for h in hits:
        for j in range(h["contact_idx"] - half, h["contact_idx"] + half + 1):
            flash[j] = h["peak_speed"]

    cap = cv2.VideoCapture(video_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h_))

    json.dump({"fps": fps, "width": w, "height": h_}, open(meta_path, "w"))

    recent = deque(maxlen=trail)
    f = 0
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "time_s", "arm_present", "wrist_x", "wrist_y",
                     "wrist_score", "wrist_speed"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            present = series["present"][f]
            if present:
                wx, wy = series["wx"][f], series["wy"][f]
                recent.append((int(wx), int(wy)))
                el = series["elbow"][f]
                if el is not None:
                    cv2.line(frame, (int(el[0]), int(el[1])), (int(wx), int(wy)),
                             ARM_COLOR, 3)
                    cv2.circle(frame, (int(el[0]), int(el[1])), 4, ARM_COLOR, -1)
                cv2.circle(frame, (int(wx), int(wy)), 7, ARM_COLOR, -1)
                sp = speed[f]
                if np.isfinite(sp):
                    cv2.putText(frame, f"{sp:.0f} px/s", (int(wx) + 10, int(wy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ARM_COLOR, 2)
                wr.writerow([f, f"{f/fps:.4f}", 1, f"{wx:.1f}", f"{wy:.1f}",
                             f"{series['score'][f]:.3f}",
                             f"{sp:.1f}" if np.isfinite(sp) else ""])
            else:
                wr.writerow([f, f"{f/fps:.4f}", 0, "", "", "", ""])

            for k, (px, py) in enumerate(recent):
                fade = (k + 1) / len(recent)
                cv2.circle(frame, (px, py), max(2, int(7 * fade)),
                           (0, int(255 * fade), 255), 2)

            if f in flash:
                cv2.rectangle(frame, (0, 0), (w - 1, h_ - 1), HIT_COLOR, 12)
                cv2.putText(frame, f"HIT  {flash[f]:.0f} px/s", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, HIT_COLOR, 4)

            out.write(frame)
            f += 1
            if f % 600 == 0:
                print(f"  [pass 2] {f} frames rendered")
    cap.release()
    out.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("video", help="input video file")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--box-conf", type=float, default=0.2,
                    help="YOLO box confidence (arm-only detections sit ~0.2-0.85)")
    ap.add_argument("--min-peak-speed", type=float, default=900.0,
                    help="min wrist peak speed (px/s) for a burst to count as a hit")
    ap.add_argument("--save-dir", default=os.path.join(os.path.dirname(__file__), "output"))
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f"{name}_drill.csv")
    video_path = os.path.join(args.save_dir, f"{name}_drill.mp4")
    meta_path = os.path.join(args.save_dir, f"{name}_drill_meta.json")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    tracker = PlayerTracker(model=args.model)
    print("[pass 1/2] tracking racket arm (YOLO11-pose) ...")
    series = collect(args.video, tracker, args.box_conf)
    n_arm = int(series["present"].sum())
    print(f"[pass 1/2] arm in frame for {n_arm}/{len(series['present'])} frames")

    data = {"t": np.arange(len(series["present"])) / fps, "wx": series["wx"],
            "wy": series["wy"], "present": series["present"], "n": len(series["wx"])}
    hits, _ = detect_hits(data, fps, min_peak_speed=args.min_peak_speed)
    print(f"[hits] {len(hits)} detected")

    print("[pass 2/2] rendering ...")
    render(args.video, series, hits, fps, csv_path, video_path, meta_path)
    print(f"csv:   {csv_path}")
    print(f"video: {video_path}")
    print(f"\nnow run:  .venv/bin/python analyze_drill.py {csv_path}")


if __name__ == "__main__":
    main()
