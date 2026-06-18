"""
scan_full.py -- sweep the whole video for fast-object (shuttle) events
======================================================================

Runs fastshuttle.scan over the full video in chunks, clusters the fast-object
tracks into timed events, gates them by travel+speed, and writes the top
candidates to output/_shuttle_candidates.json for inspection / rendering.

    .venv/bin/python scan_full.py <video>
"""

import os
import sys
import json

import cv2

import fastshuttle as fs

CHUNK = 1500


def main():
    video = sys.argv[1]
    save_dir = os.path.join(os.path.dirname(__file__), "output")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"scanning {total} frames @ {fps:.2f}fps in {CHUNK}-frame chunks")

    all_tracks = []
    for s in range(0, total, CHUNK):
        e = min(total, s + CHUNK)
        ts = fs.scan(video, start=s, end=e)
        all_tracks.extend(ts)
        print(f"  {s}-{e}: {len(ts)} tracks (cum {len(all_tracks)})")

    all_tracks.sort(key=lambda t: t["f0"])
    events = []
    for t in all_tracks:
        if events and t["f0"] <= events[-1]["f1"] + 10:
            ev = events[-1]
            ev["f1"] = max(ev["f1"], t["f1"])
            ev["tracks"].append(t)
        else:
            events.append({"f0": t["f0"], "f1": t["f1"], "tracks": [t]})

    cands = []
    for ev in events:
        best = max(ev["tracks"], key=lambda t: t["disp"] * t["peak_speed"])
        if best["disp"] < 250 or best["peak_speed"] < 2500:
            continue
        cands.append({
            "f0": ev["f0"], "f1": ev["f1"],
            "t0": round(ev["f0"] / fps, 2), "t1": round(ev["f1"] / fps, 2),
            "n_tracks": len(ev["tracks"]),
            "disp": round(best["disp"]), "peak": round(best["peak_speed"]),
            "area": round(best["mean_area"]), "straight": round(best["straightness"], 2),
            "from": [round(best["pts"][0][1]), round(best["pts"][0][2])],
            "to": [round(best["pts"][-1][1]), round(best["pts"][-1][2])],
            "score": round(best["disp"] * best["peak_speed"]),
        })
    cands.sort(key=lambda c: c["score"], reverse=True)

    out = os.path.join(save_dir, "_shuttle_candidates.json")
    json.dump(cands, open(out, "w"), indent=2)
    print(f"\n{len(all_tracks)} tracks -> {len(events)} events -> {len(cands)} candidates")
    print(f"top candidates (t0, disp, peak px/s, area, from->to):")
    for c in cands[:20]:
        print(f"  t={c['t0']:6.2f}s  disp={c['disp']:4d}  peak={c['peak']:6d}  "
              f"area={c['area']:4d}  {c['from']}->{c['to']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
