#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Record from, or play back to, the traffic counter over SDS in one command.

    python traffic/sds_session.py record 20        # 20 s of CameraIn + Detections
    python traffic/sds_session.py play             # CameraIn.<n>.sds back into the detector
    python traffic/sds_session.py play --panel     # ... and what the panel shows (Panel.<n>.p.sds)

It starts SDSIO-Server (ARM::SDS utilities) on the board's User USB
(--transport usb, the default: the firmware enumerates as "SDSIO-Client")
and drives its keyboard interface through a pseudo terminal: R starts a
recording, P a playback, S stops, X ends the server. The firmware must be
running (it opens the streams when the server says so, rec_play.c). Files go
to recordings/traffic/ (CameraIn.<n>.sds, Detections.<n>.sds, and
Detections.<n>.p.sds for a playback), next to their *.sds.yml metadata.

--panel sets the server's flag 0 (key A) before the start: the firmware then
also streams every frame as the panel shows it (Panel.<n>.sds, or
Panel.<n>.p.sds in a playback; 1.15 MB a frame, so the loop slows to what the
USB moves). traffic/panel_to_video.py makes an MP4 of it.

With --transport rtt it starts tools/sdsio_rtt_bridge.py (the J-Link's RTT
channel 1 on a TCP socket) and SDSIO-Server in connect mode on it, for a
firmware built with the SDS:IO:RTT layer. The J-Link connect halts the core;
the bridge resumes it once SDSIO-Server is connected. Stop any debug session
first if the J-Link refuses a second connection. RTT moves about one frame
per 5 s, USB the whole 40-frame clip in 8 s.
"""

from __future__ import annotations

import argparse
import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SDS_UTILITIES = Path.home() / ".cache/arm/packs/ARM/SDS/3.1.0/utilities"


def rtt_address(map_file: Path) -> int:
    """The _SEGGER_RTT control block from the linker map (the J-Link's search does not cover the DTCM)."""
    for line in map_file.read_text(errors="replace").splitlines():
        parts = line.split()
        # armlink memory map: "0x20001c34   0x000000a8   Zero   RW   5138    .bss._SEGGER_RTT    SEGGER_RTT.o"
        if ".bss._SEGGER_RTT" in parts and parts[0].startswith("0x"):
            return int(parts[0], 16)
    sys.exit(f"{map_file}: no _SEGGER_RTT")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "play"])
    ap.add_argument("seconds", nargs="?", type=float, default=20.0, help="recording length (record only)")
    ap.add_argument("--device", default="AE722F80F55D5LS_M55_HP", help="J-Link device (the AppKit-E7)")
    ap.add_argument("--map", type=Path, default=ROOT / "out/traffic/AppKit-E7/Release/traffic.axf.map")
    ap.add_argument("--workdir", type=Path, default=ROOT / "recordings/traffic")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--transport", choices=["usb", "rtt"], default="usb",
                    help="usb: the board's User USB (SDS:IO:USB, no J-Link); rtt: the J-Link RTT bridge")
    ap.add_argument("--timeout", type=float, default=1800.0, help="longest playback in seconds")
    ap.add_argument("--connect-timeout", type=float, default=120.0,
                    help="how long to wait for the board's SDSIO-Client (it connects when the firmware starts)")
    ap.add_argument("--panel", action="store_true",
                    help="also record the panel (the Panel stream, server flag 0)")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    bridge = None if args.transport == "usb" else subprocess.Popen(
        [python, str(ROOT / "tools/sdsio_rtt_bridge.py"), "--device", args.device,
         "--rtt-addr", hex(rtt_address(args.map)), "--port", str(args.port)],
        stdout=sys.stderr, stderr=sys.stderr)
    if bridge is not None:
        time.sleep(3.0)  # the bridge listens before it attaches
    interface = ["usb"] if args.transport == "usb" else ["socket", "--port", str(args.port), "--connect"]

    master, slave = pty.openpty()
    server = subprocess.Popen(
        [python, str(SDS_UTILITIES / "sdsio-server.py"), *interface,
         "--workdir", str(args.workdir), "--no-progress-info"],
        stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)

    output = []  # everything SDSIO-Server printed

    def pump(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end and server.poll() is None:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    data = os.read(master, 4096)
                except OSError:
                    return
                text = data.decode(errors="replace")
                output.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()

    try:
        # The firmware's SDSIO client connects only while it starts ("SDSIO-Client
        # USB interface initialization failed" otherwise): start the server,
        # then reset the board. Its first flags exchange ("sdsFlags = ...")
        # shows that it is there; the USB device alone may be a client that gave up.
        deadline = time.monotonic() + args.connect_timeout
        while server.poll() is None and "sdsFlags" not in "".join(output):
            if time.monotonic() > deadline:
                print(f"\nno SDSIO-Client within {args.connect_timeout:.0f} s: reset the board while the server runs",
                      file=sys.stderr)
                break
            pump(0.5)
        pump(2.0)  # the first flags exchange
        if args.panel:
            os.write(master, b"A")  # flag 0: the firmware opens the Panel stream with the others
            pump(0.5)
        if args.mode == "record":
            os.write(master, b"R")
            pump(args.seconds)
            os.write(master, b"S")
            pump(5.0)  # the target closes its streams, the server flushes the files
        else:
            started = time.time()
            os.write(master, b"P")
            # The firmware closes its streams at the end of CameraIn (about 5 s
            # per 416x416 frame over the RTT down link); SDSIO-Server reports it.
            # The Panel stream closes after Detections.
            last = "Closed:   Panel" if args.panel else "Closed:   Detections"
            while server.poll() is None and time.time() - started < args.timeout:
                pump(1.0)
                text = "".join(output)
                if last in text or ("Closed:   Detections" in text and "Record:   Panel" not in text):
                    break
            pump(3.0)
        if args.panel:
            os.write(master, b"a")
            pump(0.5)
        os.write(master, b"X")
        pump(3.0)
    finally:
        for proc in (server, bridge):
            if proc is None:
                continue
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:
                    proc.kill()
    print("\nfiles:", *sorted(p.name for p in args.workdir.glob("*.sds")), sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
