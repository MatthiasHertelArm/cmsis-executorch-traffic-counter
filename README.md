# Traffic counter on the Alif AppKit-E7: YOLO26n on the Ethos-U55

An Alif Ensemble E7 AppKit (AK-E7-AIML, Gen 2) counts the vehicles its camera
sees. Ultralytics YOLO26n, exported with ExecuTorch and compiled by Vela, runs
whole on the Ethos-U55-256 next to the Cortex-M55 HP core and finds bicycles,
cars, motorcycles, buses and trucks in each 416x416 frame. A small tracker
follows them from frame to frame and counts each one once, per class and
direction, as it crosses a line on the picture. The panel shows the camera
picture, the boxes, the line and the tallies, and the joystick moves the line.
Over the board's User USB, SDS records the model inputs and detections, or
plays a recorded clip back into the detector instead of the camera.

| | |
|---|---|
| NPU | 81 ms per 416x416 frame (1.13 GMAC, 2.46 MB program in MRAM) |
| Loop with the camera | 11.5 fps; the camera conversion runs while the NPU works |
| SDS over USB | a 40-frame clip (20.7 MB) plays in about 8 s |

The repository is a fork of the Arm example
[cmsis-executorch](https://github.com/Arm-Examples/cmsis-executorch): one CMSIS
solution, `cmsis-executorch.csolution.yml`, with one project,
`traffic/traffic.cproject.yml`, and two target-types with the same Cortex-M55
and Ethos-U55-256:

| Target-type | Hardware | Runs |
|-------------|----------|------|
| **AppKit-E7** | Alif Ensemble E7 AppKit, M55 HP core | the traffic counter: camera, panel, joystick, SDS |
| **SSE-300-U55** | Corstone-300 on the Arm FVP | the detector and the tracker on the test image, without a board |

Both use the same AI layer, `ai_layer/`. The full technical description of
the traffic counter is [documentation/traffic-counter.md](documentation/traffic-counter.md).

## 1. Host tools

1. [VS Code](https://code.visualstudio.com/) with the extensions **Keil
   Studio Pack** (`Arm.keil-studio-pack`, CMSIS Solution 1.70.0 or newer) and
   **Python** (`ms-python.python`). Sign in with an Arm account when Keil
   Studio asks; the free Keil MDK Community license is enough. When the
   folder opens, accept the tools that the Arm Tools Environment Manager
   offers from `vcpkg-configuration.json` (CMSIS-Toolbox, Arm Compiler 6,
   CMake, Ninja). The traffic counter builds with Arm Compiler 6 only.
2. **Alif SETOOLS** from the
   [Alif software and tools page](https://alifsemi.com/support/software-tools/ensemble/)
   (login required). Set its root directory, the one with `app-gen-toc` and
   `app-write-mram`, in your VS Code user settings:

   ```json
   "alif.setools.root": "/absolute/path/to/setools"
   ```

3. **SEGGER J-Link Software** V8.42 or later from
   [segger.com](https://www.segger.com/downloads/jlink/). The board has an
   on-board J-Link.
4. Python `>=3.10,<3.14` for the test image and the tools, and for SDS the
   Python packages of SDSIO-Server (`utilities/requirements.txt` of the
   `ARM::SDS` 3.1.0 pack; the USB interface needs libusb). `ffmpeg` if you
   want to turn a video into a playback.
5. For the FVP only: the Arm FVPs come with the tools from
   `vcpkg-configuration.json` on Linux and Windows. On macOS, where Arm ships
   no FVP build, `.vscode/fvp.sh` runs the Linux FVP in Docker, so Docker has
   to be running.

## 2. Board

- **PRG USB** to the host: power, the on-board J-Link and the Secure
  Enclave's UART.
- **User USB (J1)** to the host for SDS recording and playback (optional).
- The camera module the Gen 2 kit ships with, an MT9M114 on the MIPI
  connector, and the 480x800 MIPI DSI panel.
- **SW4** on `SEUART` while SETOOLS runs. The application does not use a
  UART: its console is a log buffer in the DTCM that the debugger reads.

## 3. Project and build

```bash
git clone https://github.com/MatthiasHertelArm/cmsis-executorch-traffic-counter.git
cd cmsis-executorch-traffic-counter
./setup_venv.sh && .venv/bin/python -m pip install ultralytics
.venv/bin/python traffic/make_test_image.py
```

`make_test_image.py` writes `traffic/test_image.c`, the image the detector
runs on until the camera delivers (and the only input on the FVP). It is
generated from a COCO photo that is not ours to redistribute, so it is not in
the repository, and the build needs it. The AI layer `ai_layer/` is
committed; regenerate it only after changing `model/traffic.py`. The weights
are in the repository too (`model/yolo26n-noattn-vehicles.pt`):

```bash
cbuild setup cmsis-executorch.csolution.yml --active AppKit-E7 --packs
.venv/bin/python create_ai_layer.py cmsis-executorch.cbuild-mlops.yml
```

Open the folder in VS Code and accept the pack installation
(`PyTorch::ExecuTorch`, `AlifSemiconductor::Ensemble`, `ARM::SDS`,
`Keil::MDK-Middleware`, and `ARM::V2M_MPS3_SSE_300_BSP` for the FVP). In the CMSIS view open **Manage Solution**, choose
the target-type **AppKit-E7** and click **Apply**, then **Build**. From the
command line:

```bash
cbuild cmsis-executorch.csolution.yml --active AppKit-E7 --packs
```

The image is `out/traffic/AppKit-E7/Release/traffic.axf`.

## 4. Prepare the board once

The Secure Enclave boots the HP core from a table of contents in MRAM, which
must point at the HP core's MRAM region and carry the E7's own device
configuration (with the E8's, the camera DMA cannot reach its buffers).

1. SW4 on **SEUART**, PRG USB attached.
2. **Terminal > Run Task > Alif: Install M55_HP debug stubs (AppKit-E7,
   single core configuration)**. It installs `.alif/M55_HP_mram_cfg_e7.json`
   and `.alif/app-device-config-e7.json` into SETOOLS, selects the E7 with
   silicon revision B4 and writes the table. Choose COM port discovery (`-d`)
   the first time. If `app-write-mram` gets no answer, press the board's reset
   button while it waits.

`tools/setools_mram.py out/traffic/AppKit-E7/Release/traffic.hex` does the
same and programs the application itself through the Secure Enclave, without
the J-Link.

## 5. Run and debug

In the CMSIS view click **Run** or **Debug**: the J-Link loads the image into
MRAM and starts it. The panel shows the camera picture with the line and the
tallies after a few seconds.

- **Console:** `.venv/bin/python tools/console_log.py out/traffic/AppKit-E7/Release/traffic.axf.map`
  prints the `console_log` buffer, with the tallies every 100 frames.
- **Status:** the global `traffic_status` holds the tallies, the frame
  counters, the time of each step in microseconds, the camera state, the live
  tracks and the latest detections. Writing 1 to `traffic_reset_counts`
  zeroes the tallies.
- **The line:** horizontal through the middle by default. The joystick moves
  it (left and right a vertical line, up and down a horizontal one) and the
  centre button turns it by 90 degrees. The build-time default is
  `TRAFFIC_LINE_POS` and `TRAFFIC_LINE_VERTICAL` in `traffic/traffic.cproject.yml`.

## 6. Without a board: the Corstone-300 FVP

Choose the target-type **SSE-300-U55** in **Manage Solution**, **Build**, then
**Run** (or **Debug**, which stops at `main`). The application runs the test
image through the detector and the tracker three times, prints the
detections and ends the simulation; the output is in the terminal of the
CMSIS Run task (in the Debug Console when debugging):

```text
frame 0: 2 vehicles, detect 1203 us (NPU 163 us)
  TRUCK      0.65  x1 161 y1 233 x2 293 y2 314
  TRUCK      0.50  x1 335 y1 201 x2 415 y2 363
...
Test_result: PASS
```

From the command line:

```bash
cbuild cmsis-executorch.csolution.yml --active SSE-300-U55 --packs
FVP_Corstone_SSE-300_Ethos-U55 -f board/Corstone-300/fvp_config.txt -a out/traffic/SSE-300-U55/Release/traffic.axf
```

(`.vscode/fvp.sh` in place of the model name on macOS.) The FVP's times are
not the board's: its NPU runs in fast mode.

## 7. Record and play back (SDS)

With the application running and the User USB connected, these VS Code tasks
(Terminal > Run Task) drive SDSIO-Server through `traffic/sds_session.py`:

| Task | Does |
|------|------|
| SDS: record from the camera (USB) | records `CameraIn` (the model inputs) and `Detections` for a number of seconds into `recordings/traffic/` |
| SDS: video to CameraIn stream | turns a video into `recordings/traffic/playback/CameraIn.0.sds` with ffmpeg, letterboxed or cut to fill the square input |
| SDS: play recording to the board (USB) | plays `CameraIn.<n>.sds` of a folder into the detector instead of the camera, records the board's detections next to it |
| SDS: check playback against the host model | lists the board's detections next to the float model's on the same inputs |

`traffic/count_playback.py <folder> [--sweep]` runs the tracker's logic on
the PC over the board's detections of a playback and over the float model's,
and shows which line positions the traffic actually crosses. One SDSIO server
owns the USB device at a time: let a run finish or stop it with Ctrl+C
before starting the next.

## 8. If something does not work

- **The application never gets past the Secure Enclave handshake** (the log
  shows no frames, `SRAM1: power request failed`): the Secure Enclave has
  stopped answering. A reset does not help; power-cycle the board (unplug PRG
  USB), wait about 40 s, and load again.
- **SETOOLS says "Revision is invalid":** `tools-config -p` reset the
  revision to A0. The AppKit-E7 is B4: the task and `tools/setools_mram.py`
  pass `-r B4`.
- **The build fails in CMake** (`toolchain.cmake ... could not find requested
  file`) after a CMSIS Solution extension update: delete `tmp/AppKit-E7` and
  build again.
- **The debugger cannot start its GDB server:** another debug session holds
  the port. Stop it, or change the ports in the launch configuration.
- **The first inference after a debugger reset reports error 35:** the NPU
  was still busy with the job from before the reset. The loop recovers on its
  own.
- **The picture never changes, or the board and its debug port die while the
  camera runs:** the table of contents carries another device's
  configuration. Repeat step 4. Keep the camera frames and the panel buffers
  in SRAM0: neither the camera DMA nor the display controller reaches SRAM1,
  and SRAM8 hangs the bus.

## Licences

The code is under the Apache License 2.0 ([LICENSE](LICENSE)). The YOLO26n
weights come from Ultralytics and are under the AGPL-3.0: the fine-tuned
checkpoint `model/yolo26n-noattn-vehicles.pt`, its architecture
`model/yolo26n-noattn.yaml`, and the program compiled into `ai_layer/`;
`model/` fetches the COCO-pretrained `yolo26n.pt` with the `ultralytics`
package. `src/LICENSE-ExecuTorch` covers the ExecuTorch sources
(BSD-3-Clause).

## Read on

- [documentation/traffic-counter.md](documentation/traffic-counter.md): the
  traffic counter in detail (model, training, pipeline, the FVP, memory map,
  camera bring-up on the E7, SDS over USB, NPU performance and its PMU
  counters).
- [documentation/mlops-flow.md](documentation/mlops-flow.md): the `mlops:`
  node, `*.cbuild-mlops.yml` and how the AI layer is made.
- [documentation/pack-provenance.md](documentation/pack-provenance.md): where
  the `PyTorch::ExecuTorch` pack comes from.
