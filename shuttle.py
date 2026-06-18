"""
shuttle.py -- shuttle detection via pretrained TrackNetV3 (vendored)
====================================================================

This is the shuttle half of rally-coach. It is the proven MonoTrack /
TrackNetV2 recipe, lifted from the standalone tracker:

    video frames --(TrackNetV3, pretrained)--> per-frame heatmaps
                 --(localize.fix_point)-------> one (u, v) shuttle point

The network is the stock pretrained TrackNetV3 checkpoint vendored under
./TrackNetV3 (trained on broadcast badminton). Localization is decoupled in
localize.py (paper section 4.3), so the network stays swappable.

`detect_all()` returns a dict {frame_id: result}, where result is the
localize.fix_point dict in SOURCE pixel coordinates. The combined pipeline
(track_combined.py) consumes that alongside per-frame player detections.

Deps: torch, opencv, numpy (+ the vendored TrackNetV3 repo for the model)
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "TrackNetV3"))
from dataset import Video_IterableDataset          # noqa: E402
from utils.general import get_model, WIDTH, HEIGHT  # noqa: E402  (model input 512x288)

from localize import fix_point                      # noqa: E402

DEFAULT_CKPT = os.path.join(_HERE, "TrackNetV3", "ckpts", "TrackNet_best.pt")


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tracknet(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    seq_len = ckpt["param_dict"]["seq_len"]
    bg_mode = ckpt["param_dict"]["bg_mode"]
    model = get_model("TrackNet", seq_len, bg_mode).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, seq_len, bg_mode


def track(video_file, ckpt=DEFAULT_CKPT, batch_size=4, thresh=0.3, min_area=2):
    """Yield one dict per frame: {frame, detected, uv, area, score} in src pixels."""
    device = pick_device()
    model, seq_len, bg_mode = load_tracknet(ckpt, device)
    print(f"[shuttle] device={device.type}  seq_len={seq_len}  bg_mode={bg_mode!r}")

    dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=seq_len,
                                    bg_mode=bg_mode)
    loader = DataLoader(dataset, batch_size=batch_size)
    print(f"[shuttle] {dataset.video_len} frames @ {dataset.fps} fps, {dataset.w}x{dataset.h}")

    w_scale, h_scale = dataset.w / WIDTH, dataset.h / HEIGHT
    last_emitted = -1
    for indices, x in loader:
        with torch.no_grad():
            y = model(x.float().to(device)).cpu().numpy()  # (N, L, 288, 512) in [0,1]
        for n in range(y.shape[0]):
            for f in range(y.shape[1]):
                frame_id = int(indices[n][f][1])
                if frame_id <= last_emitted:   # tail padding repeats the last frame
                    continue
                last_emitted = frame_id
                r = fix_point(y[n, f], thresh=thresh, min_area=min_area)
                if r["detected"]:
                    u, v = r["uv"]
                    r["uv"] = (u * w_scale, v * h_scale)  # back to source pixels
                yield {"frame": frame_id, **r}


def detect_all(video_file, ckpt=DEFAULT_CKPT, batch_size=4, thresh=0.3, min_area=2):
    """Run the whole video and return {frame_id: result_dict}."""
    return {r["frame"]: r for r in track(video_file, ckpt, batch_size, thresh, min_area)}
