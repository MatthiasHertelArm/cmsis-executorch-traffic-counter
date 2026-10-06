#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Bridge SEGGER RTT channel 1 of the board to a TCP socket for SDSIO-Server.

J-Link Commander's built-in RTT telnet server (port 19021) polls slowly and
dropped bytes on this board, so this script drives the J-Link DLL through
pylink directly: it polls the RTT up buffer in a tight loop and forwards the
bytes to one TCP client, and writes what the client sends into the RTT down
buffer. Run SDSIO-Server against it in connect mode:

    python tools/sdsio_rtt_bridge.py --device AE722F80F55D5LS_M55_HP --rtt-addr <_SEGGER_RTT>
    python sdsio-server.py socket --port 5050 --connect --workdir recordings

The target should be halted at its reset vector when the bridge starts;
the bridge resumes it once SDSIO-Server is connected, so that the application's
sdsOpen (5 s timeout) finds the server. --no-restart skips the resume.
"""

from __future__ import annotations

import argparse
import os
import select
import shutil
import socket
import sys
import time

import pylink


def jlink_library(path: str | None) -> pylink.Library | None:
    """The J-Link DLL next to the JLinkExe on PATH: pylink's default pick can be an older install that does not know the device."""
    if path is None:
        exe = shutil.which("JLinkExe")
        if exe is None:
            return None
        path = os.path.join(os.path.dirname(os.path.realpath(exe)), "libjlinkarm.dylib" if sys.platform == "darwin" else "libjlinkarm.so")
    return pylink.Library(dllpath=path) if os.path.exists(path) else None


def log(msg: str) -> None:
    print(f"[rtt-bridge] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="J-Link device name, e.g. AE722F80F55D5LS_M55_HP")
    ap.add_argument("--rtt-addr", type=lambda x: int(x, 0), required=True, help="address of _SEGGER_RTT (linker map)")
    ap.add_argument("--channel", type=int, default=1, help="RTT up/down buffer index used by SDSIO (default 1)")
    ap.add_argument("--port", type=int, default=5050, help="TCP port for SDSIO-Server (default 5050)")
    ap.add_argument("--speed", type=int, default=4000, help="SWD clock in kHz")
    ap.add_argument("--no-restart", action="store_true", help="do not resume the CPU when the client connects")
    ap.add_argument("--jlink-lib", help="path of libjlinkarm (default: next to the JLinkExe on PATH)")
    args = ap.parse_args()

    # Listen before attaching: the J-Link DLL opens its own servers on 19020
    # and up once connected.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(1)

    jlink = pylink.JLink(lib=jlink_library(args.jlink_lib))
    log(f"J-Link DLL {jlink._library._path} V{jlink.version}")
    jlink.open()
    jlink.exec_command("SetBatchMode 1")  # no GUI dialogs
    jlink.set_tif(pylink.enums.JLinkInterfaces.SWD)
    jlink.connect(args.device, speed=args.speed)
    log(f"connected to {jlink.core_name()}, CPU halted: {jlink.halted()}")
    log(f"waiting for SDSIO-Server on 127.0.0.1:{args.port} ...")
    conn, _ = server.accept()
    conn.setblocking(False)
    log("SDSIO-Server connected")

    if not args.no_restart and jlink.halted():
        jlink.restart()
        log("CPU resumed")

    jlink.rtt_start(args.rtt_addr)
    while True:  # the control block appears once the target has run sdsioInit
        try:
            if jlink.rtt_get_num_up_buffers() > args.channel:
                break
        except pylink.errors.JLinkRTTException:
            pass
        time.sleep(0.05)
    log("RTT control block found")

    up_total = down_total = 0
    t_report = time.monotonic()
    up_report = 0
    pending: list[int] = []  # server bytes not yet in the RTT down buffer
    t_stalled = None
    try:
        while True:
            busy = False
            data = jlink.rtt_read(args.channel, 65536)
            if data:
                conn.sendall(bytes(data))
                up_total += len(data)
                up_report += len(data)
                busy = True
            readable, _, _ = select.select([conn], [], [], 0)
            if readable:
                chunk = conn.recv(4096)
                if not chunk:
                    log("SDSIO-Server disconnected")
                    break
                pending.extend(chunk)
                busy = True
            if pending:
                # The target drains its down buffer only in sdsExchange /
                # while waiting for an answer; never spin here, the up
                # direction must keep flowing. Drop after 2 s without progress.
                written = jlink.rtt_write(args.channel, pending)
                if written:
                    del pending[:written]
                    down_total += written
                    t_stalled = None
                    busy = True
                elif t_stalled is None:
                    t_stalled = time.monotonic()
                elif time.monotonic() - t_stalled > 2.0:
                    log(f"down buffer full for 2 s, dropping {len(pending)} bytes")
                    pending.clear()
                    t_stalled = None
            now = time.monotonic()
            if now - t_report >= 5.0:
                log(f"up {up_total / 1e6:.2f} MB ({up_report / (now - t_report) / 1e3:.0f} kB/s), down {down_total} B")
                t_report, up_report = now, 0
            if not busy:
                time.sleep(0.0005)
    finally:
        conn.close()
        server.close()
        jlink.rtt_stop()
        jlink.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
