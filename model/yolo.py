# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cat detection with Ultralytics YOLO26n on the Ethos-U85.

YOLO26 is NMS-free: its one-to-one head gives at most one box per object, so
what is left after the network is a threshold, no non-maximum suppression.
This module takes the COCO-pretrained YOLO26n (class 15 is "cat"), keeps the
one-to-one head and cuts the network where the Ethos-U85 stops being useful:

  detect(image)  int8   image   (1, S, S, 3)  RGB in [0, 1], interleaved as
                                              a camera or a decoder delivers
                                              it; the NCHW transpose is the
                                              first NPU operator.
                        -> box  (1, 4, N)     left, top, right, bottom
                                              distances from each anchor, in
                                              units of the anchor's stride.
                        -> cat  (1, 1, N)     sigmoid score of the cat class.

The N = (S/8)^2 + (S/16)^2 + (S/32)^2 anchors are the cells of the three
feature maps, row by row, stride 8 first. The CPU (yolo/app_yolo.cpp)
thresholds the score and turns the distances of the few survivors into boxes:
top-k and gather, the rest of the stock head, are not NPU operators.

The classification branch keeps its 80-channel body but ends in a single
output channel, the cat logit (a 1x1 conv sliced out of the COCO head). The
other 79 classes cost nothing after the last layer, and the int8 output
quantization is spent on one score only.

Calibration uses COCO128 (train2017 images, downloaded on first use into
model/.cache/), letterboxed like the application does it; yolo/eval_cats.py
measures the float and the quantized model on the cat images of COCO val2017.
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

from model import MethodSpec

CAT_CLASS = 15  # COCO index of "cat"
IMAGE_SIZE = int(os.environ.get("YOLO_IMGSZ", "416"))  # square input, a multiple of 32
STRIDES = (8, 16, 32)
LETTERBOX_FILL = 114  # the Ultralytics pad colour, also what the application pads with
CALIBRATION_IMAGES = 64

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


class YoloCat(nn.Module):
    """YOLO26n up to the raw one-to-one head outputs, cat class only; see the module docstring."""

    def __init__(self, detection_model: nn.Module, cls: int = CAT_CLASS) -> None:
        super().__init__()
        self.layers = detection_model.model[:-1]
        self.save = set(detection_model.save)
        head = detection_model.model[-1]
        self.from_ = list(head.f)
        self.box_head = head.one2one_cv2
        cls_head = copy.deepcopy(head.one2one_cv3)
        for branch in cls_head:
            full = branch[-1]
            single = nn.Conv2d(full.in_channels, 1, kernel_size=1, bias=True)
            single.weight.data = full.weight.data[cls : cls + 1].clone()
            single.bias.data = full.bias.data[cls : cls + 1].clone()
            branch[-1] = single
        self.cls_head = cls_head

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = image.permute(0, 3, 1, 2)  # NHWC -> NCHW
        saved: list[torch.Tensor | None] = []
        for m in self.layers:
            if m.f != -1:
                x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
            x = m(x)
            saved.append(x if m.i in self.save else None)
        feats = [saved[j] for j in self.from_]
        box = torch.cat([b(f).flatten(2) for b, f in zip(self.box_head, feats)], dim=2)
        cat = torch.cat([c(f).flatten(2) for c, f in zip(self.cls_head, feats)], dim=2)
        return box, torch.sigmoid(cat)


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


def decode(box: np.ndarray, score: np.ndarray, threshold: float, size: int | None = None) -> np.ndarray:
    """Head outputs of one image, (4, N) and (N,), to detections (K, 5): x1, y1, x2, y2 in input pixels, score.

    What yolo/app_yolo.cpp does on the board; YOLO26's one-to-one head needs no NMS."""
    centre, stride = anchors(size)
    keep = np.nonzero(score > threshold)[0]
    ltrb = box[:, keep].T
    c = centre[keep]
    xyxy = np.concatenate([c - ltrb[:, :2], c + ltrb[:, 2:]], 1) * stride[keep, None]
    return np.concatenate([xyxy, score[keep, None]], 1)


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


def _samples() -> list[tuple[torch.Tensor]]:
    """Letterboxed COCO128 images, the four with cats first (the score output is pinned by them)."""
    import cv2

    images = coco128_images()
    labels = CACHE / "coco128" / "labels" / "train2017"

    def has_cat(p: Path) -> bool:
        f = labels / (p.stem + ".txt")
        return f.is_file() and any(line.split()[0] == str(CAT_CLASS) for line in f.read_text().splitlines())

    ordered = sorted(images, key=lambda p: not has_cat(p))[:CALIBRATION_IMAGES]
    return [(to_input(letterbox(cv2.imread(str(p)))[0]),) for p in ordered]


def get_yolo_methods() -> list[MethodSpec]:
    return [MethodSpec("detect", YoloCat(load_detection_model()).eval(), _samples())]
