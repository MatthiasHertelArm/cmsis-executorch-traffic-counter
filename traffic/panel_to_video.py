#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Turn the traffic counter's Panel stream into an MP4 (or one frame into a PNG).

    python traffic/panel_to_video.py recordings/traffic/panel/Panel.0.p.sds -o traffic-panel.mp4
    python traffic/panel_to_video.py recordings/traffic/panel --still 100 -o frame100.png

The firmware (rec_play.c, with SDSIO-Server's flag 0 set: sds_session.py
--panel) writes every frame the panel showed as records of 40 rows of the
CDC200's 480x800 frame buffer, all with the frame's timeslot: B, G, R per
pixel, the picture turned by 180 degrees because the panel hangs upside
down. This joins the records of each frame, turns it upright and hands it to
ffmpeg. A frame cut short (the stream stopped while it was written) is
dropped.

The video runs at the pace of the timeslots: in a playback those are the
CameraIn records' (images_to_sds.py stamps them 1000/fps ms apart), so the
video plays at the clip's speed whatever rate the board managed while it
streamed 1.15 MB a frame; --fps sets another rate.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<II")  # timeslot, data size


def headers(path: Path):
    """(timeslot, size, offset of the data) of every record, read without the data."""
    with path.open("rb") as f:
        offset = 0
        while True:
            head = f.read(HEADER.size)
            if len(head) < HEADER.size:
                return
            timeslot, size = HEADER.unpack(head)
            offset += HEADER.size
            yield timeslot, size, offset
            offset += size
            f.seek(offset)


def frames(path: Path, frame_bytes: int):
    """(timeslot, frame bytes) of every complete frame; counts the dropped ones in frames.dropped."""
    frames.dropped = 0
    with path.open("rb") as f:
        parts: list[bytes] = []
        have = 0
        slot = None
        while True:
            head = f.read(HEADER.size)
            if len(head) < HEADER.size:
                break
            timeslot, size = HEADER.unpack(head)
            data = f.read(size)
            if len(data) < size:
                break
            if slot is not None and timeslot != slot:
                frames.dropped += 1  # the rest of the last frame never came
                parts, have = [], 0
            slot = timeslot
            parts.append(data)
            have += size
            if have >= frame_bytes:
                if have == frame_bytes:
                    yield timeslot, b"".join(parts)
                else:
                    frames.dropped += 1
                parts, have, slot = [], 0, None
        if parts:
            frames.dropped += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream", type=Path, help="Panel.<n>[.p].sds, or a folder with one")
    ap.add_argument("-o", "--output", type=Path, help="the MP4 (default: next to the stream) or, with --still, the PNG")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--fps", type=float, help="frame rate of the video (default: from the timeslots)")
    ap.add_argument("--start", type=int, default=0, help="first frame")
    ap.add_argument("--frames", type=int, help="number of frames (default: all)")
    ap.add_argument("--still", type=int, metavar="N", help="write frame N as a PNG instead of a video")
    ap.add_argument("--no-turn", action="store_true", help="keep the frame buffer's orientation (upside down)")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality (lower is better)")
    args = ap.parse_args()

    stream = args.stream
    if stream.is_dir():
        found = sorted(stream.glob("Panel.*.sds"))
        if not found:
            sys.exit(f"{stream}: no Panel.<n>.sds")
        stream = found[-1]
    frame_bytes = args.width * args.height * 3

    # The pace: the mean spacing of the complete frames' timeslots.
    fps = args.fps
    if fps is None:
        sizes: dict[int, int] = {}
        for timeslot, size, _ in headers(stream):
            sizes[timeslot] = sizes.get(timeslot, 0) + size
        slots = sorted(timeslot for timeslot, size in sizes.items() if size == frame_bytes)
        if len(slots) < 2:
            fps = 30.0
        else:
            fps = round(1000.0 * (len(slots) - 1) / (slots[-1] - slots[0]), 3)

    if args.still is not None:
        first, count = args.still, 1
        output = args.output or stream.with_name(f"{stream.name.split('.')[0]}-{args.still}.png")
        codec = ["-frames:v", "1"]
    else:
        first, count = args.start, args.frames
        output = args.output or stream.with_suffix(".mp4")
        codec = ["-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart"]
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{args.width}x{args.height}", "-r", str(fps), "-i", "-", *codec, str(output)],
        stdin=subprocess.PIPE)

    written = 0
    for index, (timeslot, data) in enumerate(frames(stream, frame_bytes)):
        if index < first:
            continue
        if count is not None and written >= count:
            break
        frame = np.frombuffer(data, np.uint8).reshape(args.height, args.width, 3)  # B, G, R
        if not args.no_turn:
            frame = frame[::-1, ::-1]
        ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
        written += 1
        if written % 500 == 0:
            print(f"  {written} frames", file=sys.stderr)
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        sys.exit("ffmpeg failed")
    dropped = getattr(frames, "dropped", 0)
    print(f"{output}: {written} frames at {fps:g} fps ({written / fps:.1f} s) from {stream}"
          + (f", {dropped} incomplete frame(s) dropped" if dropped else ""))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
