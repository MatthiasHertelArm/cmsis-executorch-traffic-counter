# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ultralytics YOLO26n: what the traffic counter's model (traffic.py) and its scripts share.

YOLO26 is NMS-free: its one-to-one head gives at most one box per object, so
what is left after the network is a threshold, no non-maximum suppression.
traffic.py takes YOLO26n, keeps the one-to-one head and cuts the network
where the NPU stops being useful. This module has the parts around it: the
weights (load_detection_model), the anchors of the three feature maps (the
N = (S/8)^2 + (S/16)^2 + (S/32)^2 cells, row by row, stride 8 first), the
letterbox the application uses too, and the COCO128 images (train2017,
downloaded on first use into model/.cache/) that calibrate the quantization.
"""

from __future__ import annotations

import copy
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

IMAGE_SIZE = int(os.environ.get("YOLO_IMGSZ", "416"))  # square input, a multiple of 32
STRIDES = (8, 16, 32)
LETTERBOX_FILL = 114  # the Ultralytics pad colour, also what the application pads with

CACHE = Path(__file__).resolve().parent / ".cache"
WEIGHTS = "yolo26n.pt"
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def num_anchors(size: int | None = None) -> int:
    size = size or IMAGE_SIZE
    return sum((size // s) ** 2 for s in STRIDES)


def load_detection_model(weights: str | Path | None = None) -> nn.Module:
    """The COCO-pretrained YOLO26n (or `weights`), BatchNorm folded, one-to-one head only (Ultralytics downloads the weights)."""
    from ultralytics import YOLO

    CACHE.mkdir(exist_ok=True)
    model = YOLO(str(weights or CACHE / WEIGHTS)).model
    model = copy.deepcopy(model).float().eval()
    model.model[-1].end2end = True  # fuse() then drops the one-to-many branch and keeps one2one_*
    return model.fuse(verbose=False)


def anchors(size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Anchor centres (N, 2) in grid units and strides (N,), in the order of the head outputs."""
    size = size or IMAGE_SIZE
    centres, strides = [], []
    for s in STRIDES:
        n = size // s
        ys, xs = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
        centres.append(np.stack([xs.ravel(), ys.ravel()], 1))
        strides.append(np.full(n * n, s, dtype=np.float32))
    return np.concatenate(centres).astype(np.float32), np.concatenate(strides)


def letterbox(bgr: np.ndarray, size: int | None = None) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize the longer side to `size`, pad to a square with the Ultralytics grey; RGB uint8, the scale and the (x, y) pad."""
    import cv2

    size = size or IMAGE_SIZE

    h, w = bgr.shape[:2]
    r = size / max(h, w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), LETTERBOX_FILL, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py : py + nh, px : px + nw] = resized[:, :, ::-1]
    return out, r, (px, py)


def to_input(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb).float().div(255.0).unsqueeze(0)  # (1, S, S, 3)


def coco128_images() -> list[Path]:
    root = CACHE / "coco128"
    if not root.is_dir():
        CACHE.mkdir(exist_ok=True)
        archive = CACHE / "coco128.zip"
        print(f"[yolo] downloading {COCO128_URL}")
        # The python.org interpreters on macOS come without CA certificates: use certifi's (a requests dependency).
        import ssl

        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(COCO128_URL, context=context) as response:
            archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as z:
            z.extractall(CACHE)
        archive.unlink()
    return sorted((root / "images" / "train2017").glob("*.jpg"))
