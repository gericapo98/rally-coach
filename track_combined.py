"""
track_combined.py -- track players + shuttle in one annotated video
===================================================================

Two passes over the video:

    pass 1  shuttle.detect_all()   TrackNetV3 heatmaps -> {frame: (x,y)}
    pass 2  per frame: YOLO11-pose players, then DRAW both and write out

Outputs (in --save-dir, default ./output):
    <name>_track.mp4   players' skeletons + foot dots + fading shuttle trail
    <name>_track.csv   one row per frame, shuttle + up to 2 players

The CSV is what analyze.py turns into coaching numbers, so it is the real
product; the video is for eyeballing that the tracking is sane.

    .venv/bin/python track_combined.py output/clip.mp4
"""

import os
import csv
import argparse
from collections import deque

import cv2

import shuttle as shuttle_mod
from players import PlayerTracker, SKELETON

# distinct BGR colors per player track slot
PLAYER_COLORS = [(80, 200, 80), (80, 160, 255)]  # green, orange
SHUTTLE_COLOR = (0, 255, 255)                     # yellow


def draw_player(frame, p, color):
    k = p["kpts"]
    for a, b in SKELETON:
        if k[a][2] >= 0.3 and k[b][2] >= 0.3:
            cv2.line(frame, (int(k[a][0]), int(k[a][1])),
                     (int(k[b][0]), int(k[b][1])), color, 2)
    for x, y, s in k:
        if s >= 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    x1, y1, x2, y2 = (int(v) for v in p["box"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(frame, f"P{p['id']}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if p["foot"]:
        fx, fy = int(p["foot"][0]), int(p["foot"][1])
        cv2.drawMarker(frame, (fx, fy), color, cv2.MARKER_TILTED_CROSS, 14, 2)


def run(video_file, ckpt, save_dir, model, conf, batch_size, trail, thresh):
    name = os.path.splitext(os.path.basename(video_file))[0]
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{name}_track.csv")
    video_path = os.path.join(save_dir, f"{name}_track.mp4")

    print("[pass 1/2] shuttle (TrackNetV3) ...")
    shuttle = shuttle_mod.detect_all(video_file, ckpt=ckpt, batch_size=batch_size,
                                     thresh=thresh)
    n_sh = sum(1 for r in shuttle.values() if r["detected"])
    print(f"[pass 1/2] shuttle detected in {n_sh}/{len(shuttle)} frames")

    print("[pass 2/2] players (YOLO11-pose) + render ...")
    tracker = PlayerTracker(model=model, conf=conf)
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    header = ["frame", "time_s", "shuttle_det", "sx", "sy", "sscore"]
    for n in (1, 2):
        header += [f"p{n}_id", f"p{n}_foot_x", f"p{n}_foot_y",
                   f"p{n}_lwrist_x", f"p{n}_lwrist_y",
                   f"p{n}_rwrist_x", f"p{n}_rwrist_y"]

    recent = deque(maxlen=trail)
    f_id = 0
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            row = [f_id, f"{f_id / fps:.4f}"]

            sr = shuttle.get(f_id, {"detected": False})
            if sr["detected"]:
                sx, sy = sr["uv"]
                recent.append((int(round(sx)), int(round(sy))))
                row += [1, f"{sx:.1f}", f"{sy:.1f}", f"{sr['score']:.3f}"]
            else:
                row += [0, "", "", ""]

            players = tracker.infer(frame)
            for n in range(2):
                if n < len(players):
                    p = players[n]
                    draw_player(frame, p, PLAYER_COLORS[n])
                    foot = p["foot"] or ("", "")
                    lw = p["wrists"]["l"] or ("", "")
                    rw = p["wrists"]["r"] or ("", "")
                    row += [p["id"],
                            f"{foot[0]:.1f}" if foot[0] != "" else "",
                            f"{foot[1]:.1f}" if foot[1] != "" else "",
                            f"{lw[0]:.1f}" if lw[0] != "" else "",
                            f"{lw[1]:.1f}" if lw[1] != "" else "",
                            f"{rw[0]:.1f}" if rw[0] != "" else "",
                            f"{rw[1]:.1f}" if rw[1] != "" else ""]
                else:
                    row += ["", "", "", "", "", "", ""]

            # shuttle trail (drawn last so it sits on top)
            for k, (px, py) in enumerate(recent):
                fade = (k + 1) / len(recent)
                cv2.circle(frame, (px, py), max(2, int(8 * fade)),
                           (0, int(255 * fade), 255), 2)

            out.write(frame)
            wr.writerow(row)
            f_id += 1
            if f_id % 200 == 0:
                print(f"  ... {f_id} frames")

    cap.release()
    out.release()
    print(f"csv:   {csv_path}")
    print(f"video: {video_path}")
    return csv_path, video_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("video", help="input video file")
    ap.add_argument("--ckpt", default=shuttle_mod.DEFAULT_CKPT)
    ap.add_argument("--model", default="yolo11m-pose.pt", help="YOLO pose weights")
    ap.add_argument("--conf", type=float, default=0.4, help="YOLO player confidence")
    ap.add_argument("--thresh", type=float, default=0.3,
                    help="shuttle heatmap threshold (0.5 paper default finds "
                         "nothing on this out-of-domain phone footage; 0.3 works)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--save-dir", default=os.path.join(os.path.dirname(__file__), "output"))
    ap.add_argument("--trail", type=int, default=12)
    args = ap.parse_args()
    run(args.video, args.ckpt, args.save_dir, args.model, args.conf,
        args.batch_size, args.trail, args.thresh)


if __name__ == "__main__":
    main()
