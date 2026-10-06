#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Print the console of the traffic counter on the AppKit-E7 while it runs.

    python tools/console_log.py out/traffic/AppKit-E7/Release/traffic.axf.map [--follow SECONDS]

The AppKit-E7 board layer retargets stdio to board/AppKit-E7/retarget_stdio_log.c:
the `console_log` ring buffer in the DTCM, because SW4 stays on SEUART for the
Secure Enclave tools and UART4 is then not connected to the host. This script
finds the buffer in the linker map and reads it through the J-Link (J-Link
Commander, the core keeps running). $JLINK_DEVICE names another J-Link device.
To program an image without the debugger, see tools/setools_mram.py.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

JLINK_DEVICE = os.environ.get("JLINK_DEVICE", "AE722F80F55D5LS_M55_HP")  # the AppKit-E7's HP core


def console_log_address(map_file: Path) -> int:
    # The memory map's line of the section: "0x20000000  0x0000800c  Zero  RW  3036  .bss.console_log  <object>"
    found = re.search(r"^\s*(0x[0-9a-fA-F]+)\s+0x[0-9a-fA-F]+\s+Zero\s.*\.bss\.console_log\b", map_file.read_text(errors="replace"), re.M)
    if not found:
        sys.exit(f"{map_file}: no .bss.console_log section (is the image built with retarget_stdio_log.c?)")
    return int(found.group(1), 16)


def read_memory(address: int, size: int) -> bytes:
    """Read target memory through J-Link Commander without halting the core."""
    with tempfile.TemporaryDirectory() as tmp:
        out, script = Path(tmp) / "mem.bin", Path(tmp) / "read.jlink"
        script.write_text(f"savebin {out} {address:#x} {size:#x}\nexit\n")
        result = subprocess.run(
            ["JLinkExe", "-device", JLINK_DEVICE, "-if", "swd", "-speed", "4000", "-autoconnect", "1", "-NoGui", "1",
             "-ExitOnError", "1", "-CommandFile", str(script)],
            text=True, capture_output=True)
        if not out.is_file():
            tail = [l for l in result.stdout.splitlines() if l.strip()][-3:]
            sys.exit("[devkit] J-Link could not read the target: " + " | ".join(tail))
        return out.read_bytes()


LOG_SIZE = 0x8000  # CONSOLE_LOG_SIZE of retarget_stdio_log.c


def read_log(address: int) -> tuple[int, str]:
    raw = read_memory(address, 12 + LOG_SIZE)  # one debugger session per poll
    magic, size, head = struct.unpack("<III", raw[:12])
    if magic != 0x474F4C43 or size != LOG_SIZE:
        return 0, ""
    text = raw[12:]
    if head <= size:
        text = text[:head]
    else:  # wrapped: the oldest character is at head % size
        text = text[head % size :] + text[: head % size]
    return head, text.decode("latin-1")


def log(map_file: Path, follow: float) -> None:
    address = console_log_address(map_file)
    shown, deadline = 0, time.time() + follow
    while True:
        head, text = read_log(address)
        if head > shown:
            sys.stdout.write(text[-(head - shown):] if head - shown < len(text) else text)
            sys.stdout.flush()
            shown = head
        if time.time() >= deadline:
            break
        time.sleep(2.0)
    if shown == 0:
        print(f"[devkit] console_log at {address:#x} is empty")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("map", type=Path, help="the image's linker map (traffic.axf.map)")
    parser.add_argument("--follow", type=float, default=0.0, metavar="SECONDS", help="keep reading for this long")
    args = parser.parse_args()
    if not shutil.which("JLinkExe"):
        sys.exit("JLinkExe is not on the PATH")
    log(args.map, args.follow)


if __name__ == "__main__":
    main()
