#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Fine-tune YOLO26n without its attention blocks on the COCO images with vehicles.

    python traffic/train_vehicles.py prepare          # labels, image list, images -> tmp/train_vehicles/
    python traffic/train_vehicles.py train [--epochs 50]
    python traffic/train_vehicles.py train --resume  # continue an interrupted run from its last.pt

The Ethos-U55 has no matrix multiply, so YOLO26n's two attention blocks cost
the traffic counter either 50 ms of NPU time or 10 ms of CPU time plus the
copies between three NPU methods (documentation/traffic-counter.md).
model/yolo26n-noattn.yaml replaces them with plain C3k2 blocks; the weights
of every other layer come from the COCO-trained yolo26n.pt, and this script
fine-tunes the whole network at the traffic counter's input size.

Data (prepare): the COCO 2017 train images with a vehicle (bicycle, car,
motorcycle, bus, truck), plus NEGATIVES train images without one so that the
model keeps saying "no vehicle", all with their full 80-class labels (the
head stays COCO's, model/traffic.py picks the vehicle channels). Validation:
the COCO 2017 val images with a vehicle except the first 300, which
traffic/eval_vehicles.py uses as the test set (tmp/eval_vehicles). Labels
from Ultralytics' coco2017labels.zip, the images one by one from
images.cocodataset.org.

Train: Ultralytics on this Mac's GPU (MPS), 416 x 416, batch 32 (about 64
images per second on an M4 Max, 50 epochs in 9 hours). The best weights
are copied to model/.cache/yolo26n-noattn-vehicles.pt, which model/traffic.py
then uses.
"""

from __future__ import annotations

import argparse
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "tmp" / "train_vehicles"
MODEL_YAML = ROOT / "model" / "yolo26n-noattn.yaml"
WEIGHTS = ROOT / "model" / ".cache" / "yolo26n.pt"
INSTALLED = ROOT / "model" / ".cache" / "yolo26n-noattn-vehicles.pt"  # model/traffic.py's default
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
IMAGES_URL = "http://images.cocodataset.org/{split}/{name}.jpg"
VEHICLES = {"1", "2", "3", "5", "7"}  # COCO bicycle, car, motorcycle, bus, truck
NEGATIVES = 5000
EVAL_IMAGES = 300  # the first val images with a vehicle: traffic/eval_vehicles.py's test set
IMAGE_SIZE = 416


def has_vehicle(label: Path) -> bool:
    return label.is_file() and any(line.split()[:1] and line.split()[0] in VEHICLES for line in label.read_text().splitlines())


def fetch(url: str, dest: Path) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except OSError as err:
        print(f"{url}: {err}", file=sys.stderr)
        return False
    dest.write_bytes(data)
    return True


def prepare() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    labels_zip = WORK / "coco2017labels.zip"
    if not fetch(LABELS_URL, labels_zip):
        sys.exit("could not download the COCO labels")
    coco = WORK / "coco"
    if not (coco / "labels" / "train2017").is_dir():
        with zipfile.ZipFile(labels_zip) as z:
            z.extractall(WORK)

    def split(coco_split: str) -> tuple[list[str], list[str]]:
        names = sorted(Path(line).stem for line in (coco / f"{coco_split}.txt").read_text().split())
        labels = coco / "labels" / coco_split
        with_vehicle = [n for n in names if has_vehicle(labels / f"{n}.txt")]
        wanted = set(with_vehicle)
        return with_vehicle, [n for n in names if n not in wanted]

    train_vehicle, train_other = split("train2017")
    val_vehicle, _ = split("val2017")
    random.seed(0)
    train = sorted(train_vehicle + random.sample(train_other, NEGATIVES))
    val = val_vehicle[EVAL_IMAGES:]
    print(f"train: {len(train_vehicle)} images with a vehicle + {NEGATIVES} without; val: {len(val)}")

    jobs = []
    for out_split, coco_split, names in (("train", "train2017", train), ("val", "val2017", val)):
        (WORK / "images" / out_split).mkdir(parents=True, exist_ok=True)
        (WORK / "labels" / out_split).mkdir(parents=True, exist_ok=True)
        for name in names:
            src = coco / "labels" / coco_split / f"{name}.txt"
            (WORK / "labels" / out_split / f"{name}.txt").write_text(src.read_text() if src.is_file() else "")
            jobs.append((IMAGES_URL.format(split=coco_split, name=name), WORK / "images" / out_split / f"{name}.jpg"))
    with ThreadPoolExecutor(32) as pool:
        ok = sum(pool.map(lambda job: fetch(*job), jobs))
    print(f"{ok} of {len(jobs)} images in {WORK / 'images'}")

    import ultralytics
    import yaml

    coco_yaml = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"
    names = yaml.safe_load(coco_yaml.read_text())["names"]
    (WORK / "vehicles.yaml").write_text(
        yaml.safe_dump({"path": str(WORK), "train": "images/train", "val": "images/val", "names": names})
    )


def train(epochs: int, batch: int, resume: bool) -> None:
    from ultralytics import YOLO

    if resume:
        YOLO(str(WORK / "runs" / "noattn" / "weights" / "last.pt")).train(resume=True)
        install()
        return
    model = YOLO(str(MODEL_YAML)).load(str(WEIGHTS))
    model.train(
        data=str(WORK / "vehicles.yaml"),
        imgsz=IMAGE_SIZE,
        epochs=epochs,
        batch=batch,
        device="mps",
        workers=8,
        amp=False,
        project=str(WORK / "runs"),
        name="noattn",
        exist_ok=True,
    )
    install()


def install() -> None:
    """The best weights to model/.cache/, where model/traffic.py takes them from."""
    import shutil

    best = WORK / "runs" / "noattn" / "weights" / "best.pt"
    shutil.copyfile(best, INSTALLED)
    print(f"installed {best} as {INSTALLED}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        train(args.epochs, args.batch, args.resume)


if __name__ == "__main__":
    main()
