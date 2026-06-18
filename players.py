"""
players.py -- player detection + pose + tracking (YOLO11-pose)
=============================================================

The player half of rally-coach. Ultralytics YOLO11-pose gives, per frame:
    - a person bounding box
    - 17 COCO keypoints (the skeleton)
    - a persistent track id (ByteTrack), so "player 1" stays player 1

For "learn to hit it better" the useful signals are:
    - the player's COURT POSITION, taken as the midpoint of the two ankles
      (keypoints 15, 16) -- where the feet are, not where the box center is
    - the RACKET-SIDE WRIST (we expose both wrists; the analytics picks the
      one nearest the shuttle at contact)

We keep at most the two largest people on court (singles), discarding
line-judge / background detections by area.

Model weights (yolo11*-pose.pt) auto-download on first use into the repo.
"""

import os

import numpy as np
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))

# COCO-17 keypoint indices we care about
KP = {
    "nose": 0,
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16,
}

# COCO skeleton edges for drawing
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 5), (0, 6),                           # head-to-shoulders
]


class PlayerTracker:
    def __init__(self, model="yolo11m-pose.pt", conf=0.4, max_players=2, device=None):
        weights = model if os.path.isabs(model) else os.path.join(_HERE, model)
        self.model = YOLO(weights)
        self.conf = conf
        self.max_players = max_players
        self.device = device  # None -> ultralytics auto-picks (mps/cuda/cpu)

    def infer(self, frame):
        """Track players in one BGR frame.

        Returns a list (<= max_players) of dicts:
            {id, box (x1,y1,x2,y2), conf, kpts (17,3 -> x,y,score),
             foot (x,y) court position, wrists {l,r}}
        sorted by box area descending (the on-court singles players).
        """
        res = self.model.track(frame, persist=True, conf=self.conf,
                               verbose=False, device=self.device,
                               tracker="bytetrack.yaml")[0]
        players = []
        if res.boxes is None or res.keypoints is None:
            return players

        boxes = res.boxes.xyxy.cpu().numpy()
        ids = (res.boxes.id.cpu().numpy().astype(int)
               if res.boxes.id is not None else np.arange(len(boxes)))
        confs = res.boxes.conf.cpu().numpy()
        kpts = res.keypoints.data.cpu().numpy()  # (P, 17, 3)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            k = kpts[i]
            players.append({
                "id": int(ids[i]),
                "box": (float(x1), float(y1), float(x2), float(y2)),
                "conf": float(confs[i]),
                "kpts": k,
                "foot": self._foot(k),
                "wrists": {"l": self._kp(k, "l_wrist"), "r": self._kp(k, "r_wrist")},
                "_area": float((x2 - x1) * (y2 - y1)),
            })

        players.sort(key=lambda p: p["_area"], reverse=True)
        return players[: self.max_players]

    def arm(self, frame, box_conf=0.2, kp_score=0.3):
        """Drill mode: this footage only ever shows the racket FOREARM reaching
        in to strike a hanging shuttle -- no full body. Return the single
        highest-confidence detection's wrist + elbow (whichever side is
        visible), or {'present': False} when no arm is in frame.

            {present, side ('l'/'r'), wrist (x,y), elbow (x,y)|None,
             conf (box), score (wrist kpt), kpts}
        """
        res = self.model.predict(frame, conf=box_conf, verbose=False,
                                 device=self.device)[0]
        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            return {"present": False}
        kpts = res.keypoints.data.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        best = None
        for i in range(len(kpts)):
            k = kpts[i]
            for side in ("r", "l"):  # prefer right (racket) hand on ties
                ws = k[KP[f"{side}_wrist"]][2]
                if ws < kp_score:
                    continue
                # rank by wrist keypoint score, tie-broken by box conf
                score = ws + 0.001 * confs[i]
                if best is None or score > best[0]:
                    wx, wy = k[KP[f"{side}_wrist"]][:2]
                    ex, ey, es = k[KP[f"{side}_elbow"]]
                    best = (score, {
                        "present": True, "side": side,
                        "wrist": (float(wx), float(wy)),
                        "elbow": (float(ex), float(ey)) if es >= kp_score else None,
                        "conf": float(confs[i]), "score": float(ws), "kpts": k,
                    })
        return best[1] if best else {"present": False}

    @staticmethod
    def _kp(k, name, min_score=0.3):
        x, y, s = k[KP[name]]
        return (float(x), float(y)) if s >= min_score else None

    @classmethod
    def _foot(cls, k):
        """Court position = midpoint of visible ankles (fallback: hips, then box)."""
        la, ra = cls._kp(k, "l_ankle"), cls._kp(k, "r_ankle")
        pts = [p for p in (la, ra) if p]
        if not pts:
            lh, rh = cls._kp(k, "l_hip"), cls._kp(k, "r_hip")
            pts = [p for p in (lh, rh) if p]
        if not pts:
            return None
        return (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
