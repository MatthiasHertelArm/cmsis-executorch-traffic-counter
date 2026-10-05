#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Images to a CameraIn stream for playback into the traffic counter, and the check of what it found.

    python traffic/images_to_sds.py make [--fill] [--fps N] <out dir> <image> [<image> ...]   # -> <out dir>/CameraIn.0.sds
    python traffic/images_to_sds.py check <out dir>                        # board vs host, per image

make: every image letterboxed to the model input (416x416 RGB888, as
model/traffic.py and traffic/make_test_image.py do it) or, with --fill, its
central square cut out and scaled to fill the input (a wide video loses its
sides but no rows to the bars); one record per image, time-stamped 1000/N ms
apart in the stream's 1 kHz ticks (--fps N, default 10: 100 ms, which
traffic/eval_vehicles.py expects), plus the metadata files. The board plays
the records back at these times. Play it with
`python traffic/sds_session.py play --workdir <out dir>`.

check: reads CameraIn.0.sds and the board's Detections.0.p.sds, runs the
float model on the host on the same inputs and lists both side by side.
"""

from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "model"))

MAX_DETECTIONS = 16  # detector.h
RECORD = struct.Struct("<4I" + "5fi" * MAX_DETECTIONS)  # detections_t


def records(path: Path):
    data = path.read_bytes()
    offset = 0
    while offset + 8 <= len(data):
        timeslot, size = struct.unpack_from("<II", data, offset)
        offset += 8
        yield timeslot, data[offset : offset + size]
        offset += size


def fill(bgr, size: int):
    """The central square of the image, scaled to size x size, as RGB."""
    import cv2

    h, w = bgr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    square = bgr[y0:y0 + side, x0:x0 + side]
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.cvtColor(cv2.resize(square, (size, size), interpolation=interpolation), cv2.COLOR_BGR2RGB)


def make(out: Path, images: list[Path], fill_input: bool = False, fps: float = 10.0) -> None:
    import cv2

    from traffic import IMAGE_SIZE, letterbox

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "CameraIn.0.sds", "wb") as f:
        for i, path in enumerate(images):
            bgr = cv2.imread(str(path))
            if bgr is None:
                sys.exit(f"{path}: not an image")
            if fill_input:
                rgb = fill(bgr, IMAGE_SIZE)
            else:
                rgb, _, _ = letterbox(bgr, IMAGE_SIZE)
            data = np.ascontiguousarray(rgb).tobytes()
            f.write(struct.pack("<II", round(i * 1000 / fps), len(data)))
            f.write(data)
    (out / "CameraIn.0.txt").write_text("\n".join(str(p) for p in images) + "\n")
    for meta in ("CameraIn.sds.yml", "Detections.sds.yml"):
        shutil.copy(ROOT / "recordings/traffic" / meta, out / meta)
    print(f"{out / 'CameraIn.0.sds'}: {len(images)} records of {IMAGE_SIZE}x{IMAGE_SIZE} RGB888 at {fps:g} fps")


def check(out: Path) -> None:
    import torch

    from traffic import CLASS_NAMES, IMAGE_SIZE, YoloVehicles, decode, load_detection_model

    model = YoloVehicles(load_detection_model()).eval()
    names = (out / "CameraIn.0.txt").read_text().split() if (out / "CameraIn.0.txt").exists() else []
    board = {ts: RECORD.unpack(data[: RECORD.size]) for ts, data in records(out / "Detections.0.p.sds")}
    for i, (ts, data) in enumerate(records(out / "CameraIn.0.sds")):
        rgb = np.frombuffer(data, np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
        with torch.no_grad():
            box, score = model(torch.from_numpy(rgb.copy()).float().div(255).unsqueeze(0))
        host = decode(box[0].numpy(), score[0].numpy(), 0.30)
        name = Path(names[i]).name if i < len(names) else f"#{i}"
        rec = board.get(ts)
        if rec is None:
            print(f"{name:24s} board: no result   host: {len(host)} vehicle(s)")
            continue
        count = rec[1]
        boxes = [rec[4 + 6 * k : 10 + 6 * k] for k in range(count)]
        b = "  ".join(f"{CLASS_NAMES[c]} {s:.2f}@({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})" for x1, y1, x2, y2, s, c in boxes)
        h = "  ".join(f"{CLASS_NAMES[int(d[5])]} {d[4]:.2f}@({d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f})" for d in host)
        print(f"{name:24s} board {count}: {b:40s} host {len(host)}: {h}  (NPU {rec[2]} us)")


def main() -> None:
    args = sys.argv[1:]
    fill_input = "--fill" in args
    args = [a for a in args if a != "--fill"]
    fps = 10.0
    if "--fps" in args:
        i = args.index("--fps")
        fps = float(args[i + 1])
        del args[i : i + 2]
    if len(args) >= 3 and args[0] == "make":
        make(Path(args[1]), [Path(p) for p in args[2:]], fill_input, fps)
    elif len(args) == 2 and args[0] == "check":
        check(Path(args[1]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
