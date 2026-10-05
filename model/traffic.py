# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vehicle detection with Ultralytics YOLO26n on the Ethos-U55, for the traffic counter.

The same cut of YOLO26n as model/yolo.py (the NMS-free one-to-one head, the
network up to the raw head outputs on the NPU, the threshold and the box
decode on the CPU), with the classification branch ending in the vehicle
classes of COCO instead of the cat alone:

  image  int8 (1, S, S, 3)  RGB in [0, 1], interleaved
  -> box int8 (1, N, 4)     left, top, right, bottom distances
                            from each anchor, in anchor strides
  -> cls int8 (1, N, C)     sigmoid score per class, C = len(CLASSES)

By default (TRAFFIC_ATTENTION=cpu) that is three NPU methods, stem, mid and
head, with the two attention cores on the CPU in between (see Split below);
TRAFFIC_ATTENTION=npu exports it as the single method `detect`.

The N anchors are the cells of the stride 8, 16 and 32 maps, row by row,
stride 8 first. The CPU (traffic/detector.cpp) takes the best class of each
anchor, thresholds it and turns the distances into boxes; traffic/tracker.c
follows the boxes from frame to frame and counts them across a line.

The Ethos-U55 (the E7's NPU) has no TRANSPOSE: the NHWC to NCHW permute that
the Ethos-U85 ran as its first operator is handled by the Arm backend's
layout passes, and whatever it cannot fold stays on the CPU, so the export
reports what runs where. What the layout passes cannot fold, Vela turns into
one copy per channel. So the outputs are anchor-major, (1, N, 4) and (1, N, C),
which is the NPU's NHWC order flattened (channel-major outputs cost 209 such
copies), and the two attention blocks take q, k and v as column slices of the
(1, N, C) token view (TokenMajorAttention) instead of the (1, heads, dims, N)
view Ultralytics uses (about 870 copies).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import MethodSpec
from yolo import (  # noqa: F401  (re-exported for the scripts in traffic/)
    LETTERBOX_FILL,
    STRIDES,
    anchors,
    coco128_images,
    letterbox,
    num_anchors,
    to_input,
)
import yolo
from ultralytics.nn.modules.block import PSABlock

# COCO index and name of each output channel, in order.
CLASSES = ((1, "bicycle"), (2, "car"), (3, "motorcycle"), (5, "bus"), (7, "truck"))
CLASS_INDICES = [c for c, _ in CLASSES]
CLASS_NAMES = [n for _, n in CLASSES]
IMAGE_SIZE = int(os.environ.get("YOLO_IMGSZ", "416"))  # square input, a multiple of 32
CALIBRATION_IMAGES = 64

CACHE = Path(__file__).resolve().parent / ".cache"


class TokenMajorAttention(nn.Module):
    """The attention of a YOLO26 PSABlock, the same arithmetic on the channels-last token view.

    Ultralytics views the qkv map as (B, heads, 2 * key_dim + head_dim, N) and
    splits q, k, v along the dims axis: on the NPU's NHWC tensors that view is
    a transpose. Here q, k and v of each head are column slices of the
    (B, N, channels) view of the NHWC map, which is free."""

    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        self.qkv, self.proj, self.pe = attn.qkv, attn.proj, attn.pe
        self.heads, self.key_dim, self.head_dim, self.scale = attn.num_heads, attn.key_dim, attn.head_dim, attn.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        kd = self.key_dim
        tokens = self.qkv(x).permute(0, 2, 3, 1).reshape(B, H * W, self.heads, 2 * kd + self.head_dim)
        out, values = [], []
        for h in range(self.heads):
            q = tokens[:, :, h, :kd] * self.scale
            k = tokens[:, :, h, kd : 2 * kd]
            v = tokens[:, :, h, 2 * kd :]
            out.append((q @ k.transpose(1, 2)).softmax(-1) @ v)
            values.append(v)
        o = torch.cat(out, -1).reshape(B, H, W, C).permute(0, 3, 1, 2)
        v = torch.cat(values, -1).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.proj(o + self.pe(v))


class YoloVehicles(nn.Module):
    """YOLO26n up to the raw one-to-one head outputs, the vehicle classes only; see the module docstring."""

    def __init__(self, detection_model: nn.Module, classes: list[int] | None = None) -> None:
        super().__init__()
        classes = classes or CLASS_INDICES
        self.layers = detection_model.model[:-1]
        self.save = set(detection_model.save)
        head = detection_model.model[-1]
        self.from_ = list(head.f)
        self.box_head = head.one2one_cv2
        cls_head = copy.deepcopy(head.one2one_cv3)
        for branch in cls_head:
            full = branch[-1]
            few = nn.Conv2d(full.in_channels, len(classes), kernel_size=1, bias=True)
            few.weight.data = full.weight.data[classes].clone()
            few.bias.data = full.bias.data[classes].clone()
            branch[-1] = few
        self.cls_head = cls_head
        for block in self.layers.modules():
            if isinstance(block, PSABlock):
                block.attn = TokenMajorAttention(block.attn)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = image.permute(0, 3, 1, 2)  # NHWC -> NCHW
        saved: list[torch.Tensor | None] = []
        for m in self.layers:
            if m.f != -1:
                x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
            x = m(x)
            saved.append(x if m.i in self.save else None)
        feats = [saved[j] for j in self.from_]
        # Anchor-major: each map's NHWC order flattened, (1, N, 4) and (1, N, C).
        box = torch.cat([b(f).permute(0, 2, 3, 1).flatten(1, 2) for b, f in zip(self.box_head, feats)], dim=1)
        cls = torch.cat([c(f).permute(0, 2, 3, 1).flatten(1, 2) for c, f in zip(self.cls_head, feats)], dim=1)
        return box, torch.sigmoid(cls)


def decode(box: np.ndarray, score: np.ndarray, threshold: float, size: int | None = None) -> np.ndarray:
    """Head outputs of one image, (N, 4) and (N, C), to detections (K, 6): x1, y1, x2, y2 in input pixels, score, class.

    What traffic/detector.cpp does on the board: the best class per anchor, then the threshold."""
    size = size or IMAGE_SIZE
    centre, stride = anchors(size)
    best = score.argmax(1)
    conf = score[np.arange(score.shape[0]), best]
    keep = np.nonzero(conf > threshold)[0]
    ltrb = box[keep]
    c = centre[keep]
    xyxy = np.concatenate([c - ltrb[:, :2], c + ltrb[:, 2:]], 1) * stride[keep, None]
    return np.concatenate([xyxy, conf[keep, None], best[keep, None].astype(np.float32)], 1)


# ---------------------------------------------------------------------------
# The attention on the CPU. The Ethos-U55 has no matrix multiply: Vela runs
# each of the two per-block products (q k^T and the attention-weighted sum of
# v) as one broadcast multiply with int32 results plus one REDUCE_SUM per
# row, 8 bytes of SRAM traffic per MAC, about 50 of the 80 ms of the whole
# network for 1 % of its MACs. With TRAFFIC_ATTENTION=cpu (the default) the
# network is cut around the two attention cores into three NPU methods,
#
#   stem(image)                              -> l4, l6, y10, qkv10
#   mid(o10, qkv10, y10, l4, l6)             -> qkv22, z22, y22, box34, cls34
#   head(o22, qkv22, z22, y22, box34, cls34) -> box, cls
#
# and the CPU (traffic/attention.c) turns qkv into o = softmax(q k^T s) v per
# head in between. Every tensor crosses in NHWC (the NPU's order, no
# transposes). mid already runs the stride 8 and 16 heads (box34, and cls34
# as logits); head appends the stride 32 anchors and takes the sigmoid, so
# box and cls are `detect`'s outputs, one int8 range each: a score output of
# the stride 8 + 16 anchors alone got the range of what COCO128 scores there
# (up to 0.19) and clipped every stronger detection.
# The same tensor gets different quantization parameters as an output of one
# method and an input of the next (concatenations share one range), so the
# CPU requantizes on the way. TRAFFIC_ATTENTION=npu exports the single
# `detect` method instead.
# ---------------------------------------------------------------------------
ATTENTION = os.environ.get("TRAFFIC_ATTENTION", "cpu")
# The weights: TRAFFIC_WEIGHTS, else YOLO26n without its attention blocks
# fine-tuned on the COCO images with vehicles (model/yolo26n-noattn.yaml,
# traffic/train_vehicles.py installs it as model/.cache/yolo26n-noattn-vehicles.pt),
# else the COCO-pretrained yolo26n.pt. A network without PSA blocks is always
# the single method `detect`: there is no attention to take off the NPU.
TRAINED = CACHE / "yolo26n-noattn-vehicles.pt"
WEIGHTS = os.environ.get("TRAFFIC_WEIGHTS") or (str(TRAINED) if TRAINED.is_file() else None)


def load_detection_model(weights: str | Path | None = None) -> nn.Module:
    """yolo.load_detection_model() of `weights`, TRAFFIC_WEIGHTS, the fine-tuned model or yolo26n.pt."""
    weights = weights or WEIGHTS
    print(f"[traffic] weights: {weights or yolo.CACHE / yolo.WEIGHTS}")
    return yolo.load_detection_model(weights)


def _nhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1)


def _nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)


def _anchor_major(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).flatten(1, 2)


def attention_core(qkv: torch.Tensor, attn: TokenMajorAttention) -> torch.Tensor:
    """The CPU part in float: NHWC qkv (B, H, W, heads * (2 kd + hd)) -> o (B, H, W, heads * hd)."""
    B, H, W, _ = qkv.shape
    kd, hd = attn.key_dim, attn.head_dim
    t = qkv.reshape(B, H * W, attn.heads, 2 * kd + hd)
    out = [
        ((t[:, :, h, :kd] * attn.scale) @ t[:, :, h, kd : 2 * kd].transpose(1, 2)).softmax(-1) @ t[:, :, h, 2 * kd :]
        for h in range(attn.heads)
    ]
    return torch.cat(out, -1).reshape(B, H, W, attn.heads * hd)


def _round(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)  # half away from zero, like vcvtaq / lroundf


def attention_core_int8(
    qkv: np.ndarray, qkv_qp: tuple[float, int], out_qp: tuple[float, int], attn: TokenMajorAttention
) -> np.ndarray:
    """What traffic/attention.c computes, for the host evaluation: int8 (H, W, heads * (2 kd + hd)) -> int8 (H, W, heads * hd).

    Per head and query row i: exact int32 dot products d_ij of the zero-point
    corrected q and k, e_ij = exp((d_ij - max_j d_ij) s_q s_k scale), p_ij =
    round(255 e_ij) as uint8 (relative to the row's largest weight), then
    o_ic = sum_j p_ij (v_jc - zp) / sum_j p_ij, requantized to the scale the
    next NPU method expects."""
    (s_in, zp), (s_out, zp_out) = qkv_qp, out_qp
    H, W, _ = qkv.shape
    kd, hd = attn.key_dim, attn.head_dim
    t = qkv.reshape(H * W, attn.heads, 2 * kd + hd).astype(np.int64)
    out = np.empty((H * W, attn.heads * hd), np.int8)
    c = np.float32(s_in * s_in * attn.scale)
    for h in range(attn.heads):
        q, k, v = t[:, h, :kd] - zp, t[:, h, kd : 2 * kd] - zp, t[:, h, 2 * kd :]
        d = q @ k.T
        e = np.exp(((d - d.max(1, keepdims=True)).astype(np.float32) * c))
        p = np.floor(e * 255 + 0.5).astype(np.int64)
        sp = p.sum(1, keepdims=True)
        acc = p @ (v + 128) - (128 + zp) * sp
        o = acc.astype(np.float32) * (np.float32(s_in / s_out) / sp.astype(np.float32))
        out[:, h * hd : (h + 1) * hd] = np.clip(_round(o) + zp_out, -128, 127)
    return out.reshape(H, W, attn.heads * hd)


def _finish_psa(block: nn.Module, x: torch.Tensor, o: torch.Tensor, qkv: torch.Tensor) -> torch.Tensor:
    """The rest of a PSABlock once the CPU has the attention core o: pe(v), proj, both residuals, the FFN."""
    attn = block.attn
    B, H, W, _ = o.shape
    kd = attn.key_dim
    v = qkv.reshape(B, H * W, attn.heads, 2 * kd + attn.head_dim)[..., 2 * kd :].reshape(B, H, W, -1)
    x = x + attn.proj(_nchw(o) + attn.pe(_nchw(v)))
    return x + block.ffn(x)


class Stem(nn.Module):
    """Layers 0-9 and layer 10 (C2PSA) up to its attention's qkv: image -> l4, l6, y10 (cv1 of layer 10), qkv10."""

    def __init__(self, net: YoloVehicles) -> None:
        super().__init__()
        self.net = net

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        layers = self.net.layers
        x = image.permute(0, 3, 1, 2)
        for i in range(10):
            x = layers[i](x)
            if i == 4:
                l4 = x
            elif i == 6:
                l6 = x
        c2psa = layers[10]
        y = c2psa.cv1(x)
        qkv = c2psa.m[0].attn.qkv(y[:, c2psa.c :])
        return _nhwc(l4), _nhwc(l6), _nhwc(y), _nhwc(qkv)


class Mid(nn.Module):
    """The rest of layer 10, the neck (11-21), layer 22 (C3k2 with PSA) up to its qkv, and the stride 8 and 16 heads."""

    def __init__(self, net: YoloVehicles) -> None:
        super().__init__()
        self.net = net

    def forward(self, o10, qkv10, y10, l4, l6) -> tuple[torch.Tensor, ...]:
        net = self.net
        c2psa = net.layers[10]
        y = _nchw(y10)
        b = _finish_psa(c2psa.m[0], y[:, c2psa.c :], o10, qkv10)
        x = c2psa.cv2(torch.cat((y[:, : c2psa.c], b), 1))
        saved = {4: _nchw(l4), 6: _nchw(l6), 10: x}
        for m in net.layers[11:22]:
            if m.f != -1:
                x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
            x = m(x)
            if m.i in net.save:
                saved[m.i] = x
        c3k2 = net.layers[22]
        y22 = c3k2.cv1(x)
        z = c3k2.m[0][0](y22[:, c3k2.c :])
        qkv22 = c3k2.m[0][1].attn.qkv(z)
        feats = [saved[j] for j in net.from_[:2]]
        box = torch.cat([_anchor_major(b(f)) for b, f in zip(net.box_head[:2], feats)], dim=1)
        cls = torch.cat([_anchor_major(c(f)) for c, f in zip(net.cls_head[:2], feats)], dim=1)
        return _nhwc(qkv22), _nhwc(z), _nhwc(y22), box, cls


class Head(nn.Module):
    """The rest of layer 22 and the stride 32 head, appended to mid's stride 8 and 16 anchors."""

    def __init__(self, net: YoloVehicles) -> None:
        super().__init__()
        self.net = net

    def forward(self, o22, qkv22, z22, y22, box34, cls34) -> tuple[torch.Tensor, torch.Tensor]:
        net = self.net
        c3k2 = net.layers[22]
        y = _nchw(y22)
        y2 = _finish_psa(c3k2.m[0][1], _nchw(z22), o22, qkv22)
        x = c3k2.cv2(torch.cat((y[:, : c3k2.c], y[:, c3k2.c :], y2), 1))
        box = torch.cat((box34, _anchor_major(net.box_head[2](x))), 1)
        return box, torch.sigmoid(torch.cat((cls34, _anchor_major(net.cls_head[2](x))), 1))


class Split:
    """stem -> attention core -> mid -> attention core -> head, as the board runs it; float, or quantized."""

    def __init__(self, net: YoloVehicles) -> None:
        self.stem, self.mid, self.head = Stem(net).eval(), Mid(net).eval(), Head(net).eval()
        self.attn10 = net.layers[10].m[0].attn
        self.attn22 = net.layers[22].m[0][1].attn
        self.quantized = None

    def stages(self, image: torch.Tensor) -> list[tuple[torch.Tensor, ...]]:
        """The float inputs of stem, mid and head for one image (the calibration samples)."""
        with torch.no_grad():
            l4, l6, y10, qkv10 = self.stem(image)
            mid_in = (attention_core(qkv10, self.attn10), qkv10, y10, l4, l6)
            qkv22, z22, y22, box34, cls34 = self.mid(*mid_in)
            return [(image,), mid_in, (attention_core(qkv22, self.attn22), qkv22, z22, y22, box34, cls34)]

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.quantized is not None:
            return self._int8(image)
        with torch.no_grad():
            return self.head(*self.stages(image)[2])

    def quantize(self, quantize_method) -> "Split":
        """The same pipeline with the PT2E-quantized methods and the integer core (attention_core_int8)."""
        q = Split.__new__(Split)
        q.attn10, q.attn22 = self.attn10, self.attn22
        q.quantized = {m.name: (quantize_method(m), m) for m in self.methods()}
        q.qparams = {name: io_qparams(gm) for name, (gm, _) in q.quantized.items()}
        return q

    def _int8(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stem, mid, head = (self.quantized[n][0] for n in ("stem", "mid", "head"))
        qp = self.qparams

        def core(qkv, qkv_qp, out_qp, attn):
            s, z = qkv_qp
            qkv_q = torch.clamp(torch.round(qkv / s) + z, -128, 127)[0].numpy().astype(np.int8)
            o = attention_core_int8(qkv_q, qkv_qp, out_qp, attn)
            return torch.from_numpy((o.astype(np.float32) - out_qp[1]) * out_qp[0])[None]

        with torch.no_grad():
            l4, l6, y10, qkv10 = stem(image)
            o10 = core(qkv10, qp["stem"][1][3], qp["mid"][0][0], self.attn10)
            qkv22, z22, y22, box34, cls34 = mid(o10, qkv10, y10, l4, l6)
            o22 = core(qkv22, qp["mid"][1][0], qp["head"][0][0], self.attn22)
            return head(o22, qkv22, z22, y22, box34, cls34)

    def methods(self) -> list[MethodSpec]:
        samples = [self.stages(image) for (image,) in _samples()]
        return [
            MethodSpec("stem", self.stem, [s[0] for s in samples]),
            MethodSpec("mid", self.mid, [s[1] for s in samples]),
            MethodSpec("head", self.head, [s[2] for s in samples]),
        ]


def io_qparams(gm: torch.fx.GraphModule) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    """(scale, zero point) of each input and output of a PT2E-quantized module."""
    ins, outs = [], []
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            q = next(u for u in n.users if "quantize_per_tensor" in str(u.target))
            ins.append((float(q.args[1]), int(q.args[2])))
        elif n.op == "output":
            outs = [(float(d.args[1]), int(d.args[2])) for d in n.args[0]]
    return ins, outs


def _samples() -> list[tuple[torch.Tensor]]:
    """Letterboxed COCO128 images, those with vehicles first (they pin the score output's range)."""
    import cv2

    images = coco128_images()
    labels = CACHE / "coco128" / "labels" / "train2017"
    wanted = {str(c) for c in CLASS_INDICES}

    def vehicles(p: Path) -> int:
        f = labels / (p.stem + ".txt")
        return sum(line.split()[0] in wanted for line in f.read_text().splitlines()) if f.is_file() else 0

    ordered = sorted(images, key=lambda p: -vehicles(p))[:CALIBRATION_IMAGES]
    return [(to_input(letterbox(cv2.imread(str(p)), IMAGE_SIZE)[0]),) for p in ordered]


def _has_attention(net: YoloVehicles) -> bool:
    return any(isinstance(m, TokenMajorAttention) for m in net.modules())


def get_traffic_methods() -> list[MethodSpec]:
    yolo.IMAGE_SIZE = IMAGE_SIZE
    net = YoloVehicles(load_detection_model()).eval()
    if ATTENTION == "npu" or not _has_attention(net):
        return [MethodSpec("detect", net, _samples())]
    return Split(net).methods()


def runners(quantize_method) -> tuple:
    """The float and the int8 detector as the board runs it, image -> (box (1, N, 4), score (1, N, C)).

    `quantize_method(spec)` is create_ai_layer.quantize_method bound to the
    compile spec; the int8 one is its fake quantization on the host (plus the
    integer attention core with TRAFFIC_ATTENTION=cpu)."""
    yolo.IMAGE_SIZE = IMAGE_SIZE
    net = YoloVehicles(load_detection_model()).eval()
    if ATTENTION == "npu" or not _has_attention(net):
        return net, quantize_method(MethodSpec("detect", net, _samples()))
    split = Split(net)
    return split, split.quantize(quantize_method)
