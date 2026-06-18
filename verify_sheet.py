"""verify_sheet.py -- one contact sheet of the top candidate events for eyeballing."""
import os, json, cv2, numpy as np
import fastshuttle as fs

video = os.path.expanduser("~/Downloads/20260529_195739.mp4")
HERE = os.path.dirname(__file__)
cands = json.load(open(os.path.join(HERE, "output/_shuttle_candidates.json")))
TOPN = int(os.environ.get("TOPN", "15"))
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

rows = []
for c in cands[:TOPN]:
    f0, f1 = c["f0"], c["f1"]
    tr = fs.scan(video, start=max(0, f0 - 3), end=f1 + 4)
    top = tr[0] if tr else None
    poly = [(p[0], int(p[1]), int(p[2])) for p in top["pts"]] if top else []
    span = [f0, (f0 + f1) // 2, f1]
    tiles = []
    for fid in span:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, fr = cap.read()
        if not ok:
            fr = np.zeros((1920, 1080, 3), np.uint8)
        for k in range(1, len(poly)):
            cv2.line(fr, poly[k-1][1:], poly[k][1:], (0, 180, 255), 3)
        cur = [p for p in poly if p[0] == fid]
        if cur:
            cv2.circle(fr, cur[0][1:], 24, (0, 255, 255), 5)
        tiles.append(cv2.resize(fr, (0, 0), fx=0.16, fy=0.16))
    row = np.hstack(tiles)
    cv2.putText(row, f"t={c['t0']:.1f}s area={c['area']} peak={c['peak']} disp={c['disp']} {c['from']}->{c['to']}",
                (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    rows.append(row)
cap.release()
W = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
sheet = np.vstack(rows)
out = os.path.join(HERE, "output/_verify_sheet.png")
cv2.imwrite(out, sheet)
print("wrote", out, sheet.shape)
