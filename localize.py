"""
localize.py  --  "fix a single point to the shuttlecock"
========================================================

This is the final localization step of the MonoTrack / TrackNetV2 detector
(paper section 4.3), implemented exactly as described:

    1. the network outputs a per-frame heatmap  H in [0,1]
    2. threshold H at 0.5  -> binary mask
    3. take the centroid of the LARGEST connected component
    4. if no pixels clear the threshold, or the blob is too small,
       report the shuttle as UNDETECTED

It is deliberately decoupled from the network: feed it ANY heatmap (pretrained
TrackNet, a fine-tuned model, or later your own) and it returns one (u, v) point.

`subpixel=True` adds an intensity-weighted centroid inside the winning blob,
which is a small, safe improvement over the paper's binary centroid and matters
when the shuttle is only a few pixels wide.

Deps: numpy, opencv (cv2)
"""

import numpy as np
import cv2


def fix_point(heatmap, thresh=0.5, min_area=3, subpixel=True):
    """Collapse a heatmap to a single shuttle point.

    Parameters
    ----------
    heatmap : 2D float array in [0, 1]   (the network output)
    thresh  : detection threshold (paper uses 0.5)
    min_area: reject blobs smaller than this many pixels -> 'undetected'
    subpixel: intensity-weighted centroid inside the winning blob

    Returns
    -------
    dict with:
        detected : bool
        uv       : (u, v) pixel point or None
        area     : pixels in the winning blob
        score    : mean heatmap value inside the blob (a confidence)
    """
    H = np.asarray(heatmap, dtype=np.float32)
    mask = (H >= thresh).astype(np.uint8)

    # connected components; label 0 is background
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:                                   # nothing above threshold
        return {"detected": False, "uv": None, "area": 0, "score": 0.0}

    # largest component by area (skip background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    area = int(stats[best, cv2.CC_STAT_AREA])
    if area < min_area:                          # blob too small -> undetected
        return {"detected": False, "uv": None, "area": area, "score": 0.0}

    blob = (labels == best)
    score = float(H[blob].mean())

    if subpixel:
        ys, xs = np.where(blob)
        w = H[ys, xs]                            # weight by heatmap intensity
        u = float((xs * w).sum() / w.sum())
        v = float((ys * w).sum() / w.sum())
    else:
        u, v = float(centroids[best][0]), float(centroids[best][1])

    return {"detected": True, "uv": (u, v), "area": area, "score": score}


# --------------------------------------------------------------------------
# demo: prove the rule on synthetic heatmaps (no network needed)
# --------------------------------------------------------------------------
def _gaussian_blob(shape, center, sigma, peak):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    g = peak * np.exp(-(((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
                        / (2 * sigma ** 2)))
    return g


def _demo():
    H, W = 120, 200
    rng = np.random.default_rng(0)

    # case 1: a clear shuttle blob + a weaker distractor (e.g. a reflection)
    hm = _gaussian_blob((H, W), center=(130, 60), sigma=2.5, peak=0.95)
    hm += _gaussian_blob((H, W), center=(40, 90), sigma=4.0, peak=0.45)  # distractor
    hm += rng.normal(0, 0.02, (H, W))
    hm = np.clip(hm, 0, 1)
    r = fix_point(hm)
    print("case 1 (shuttle at 130,60 + weak distractor):")
    print(f"   -> {r}   [should lock near (130, 60)]")

    # case 2: nothing clears 0.5 -> must report undetected (white-on-white frame)
    hm2 = np.clip(_gaussian_blob((H, W), (100, 50), 3.0, 0.30)
                  + rng.normal(0, 0.02, (H, W)), 0, 1)
    r2 = fix_point(hm2)
    print("case 2 (faint blob, peak 0.30):")
    print(f"   -> {r2}   [should be detected=False]")

    # case 3: two equally strong blobs (cork-flip ambiguity at the heatmap level)
    hm3 = _gaussian_blob((H, W), (90, 60), 2.5, 0.9)
    hm3 += _gaussian_blob((H, W), (110, 60), 2.5, 0.9)
    hm3 = np.clip(hm3, 0, 1)
    r3 = fix_point(hm3)
    print("case 3 (two equal blobs, the cork-flip case):")
    print(f"   -> {r3}   [note: localizer alone merges/averages; the FIX for")
    print("       cork-flip is the trajectory gate at the track level, not here]")


if __name__ == "__main__":
    _demo()
