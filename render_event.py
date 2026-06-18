"""
render_event.py -- montage one candidate shuttle event for inspection
=====================================================================

Given a frame window, re-scan it for the best fast-object track and draw that
track over a few sampled source frames, tiled into one image. Used by the
verify agents (to judge shuttle vs arm vs noise) and by the final render.

    .venv/bin/python render_event.py <video> <f0> <f1> <out.png>

Prints a JSON summary of the best track to stdout.
"""

import sys
import json

import cv2
import numpy as np

import fastshuttle as fs


def main():
    video, f0, f1, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    pad = 4
    tracks = fs.scan(video, start=max(0, f0 - pad), end=f1 + pad)
    top = tracks[0] if tracks else None

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    # sample up to 8 frames across the window
    span = list(range(f0, f1 + 1))
    if len(span) > 8:
        idx = np.linspace(0, len(span) - 1, 8).astype(int)
        span = [span[i] for i in idx]

    pts = {p[0]: (p[1], p[2]) for p in top["pts"]} if top else {}
    allpts = [(p[0], p[1], p[2]) for p in top["pts"]] if top else []
    tiles = []
    for fid in span:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, fr = cap.read()
        if not ok:
            continue
        # draw the whole track polyline faintly, then the point at this frame
        for k in range(1, len(allpts)):
            cv2.line(fr, (int(allpts[k - 1][1]), int(allpts[k - 1][2])),
                     (int(allpts[k][1]), int(allpts[k][2])), (0, 180, 255), 2)
        if fid in pts:
            cv2.circle(fr, (int(pts[fid][0]), int(pts[fid][1])), 22, (0, 255, 255), 4)
        cv2.putText(fr, f"f{fid} t={fid/fps:.2f}s", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        tiles.append(cv2.resize(fr, (0, 0), fx=0.22, fy=0.22))
    cap.release()
    if tiles:
        # 2 rows
        per = (len(tiles) + 1) // 2
        rows = []
        for r in range(0, len(tiles), per):
            row = tiles[r:r + per]
            while len(row) < per:
                row.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(row))
        cv2.imwrite(out, np.vstack(rows))

    summary = {"event_f0": f0, "event_f1": f1, "t0": round(f0 / fps, 2),
               "t1": round(f1 / fps, 2), "image": out,
               "best_track": None if not top else {
                   "disp": round(top["disp"]), "peak_speed": round(top["peak_speed"]),
                   "mean_area": round(top["mean_area"]), "n": top["n"],
                   "straightness": round(top["straightness"], 2),
                   "from": [round(top["pts"][0][1]), round(top["pts"][0][2])],
                   "to": [round(top["pts"][-1][1]), round(top["pts"][-1][2])]}}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
