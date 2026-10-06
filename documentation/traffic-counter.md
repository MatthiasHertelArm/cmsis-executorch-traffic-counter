# Traffic counter with YOLO26n on the Ethos-U55 (AppKit-E7)

The vehicles an Alif Ensemble E7 AppKit (AK-E7-AIML, Gen 2) sees through its
camera are detected by Ultralytics YOLO26n on the Ethos-U55-256 next to the
Cortex-M55 HP core, followed from frame to frame by a small tracker and
counted, per class and direction, as they cross a line on the picture. The
panel shows the picture, the tracks, the line and the tallies. The project is
`traffic/traffic.cproject.yml`, target-type `AppKit-E7`, a fork of the cat
detector of the AppKit-E8 ([yolo-cats.md](yolo-cats.md)).

## The model

[`model/traffic.py`](../model/traffic.py) is the cut of YOLO26n that
`model/yolo.py` makes for the cat, with the classification branch ending in
the five COCO vehicle classes instead of the cat alone. Its outputs are
anchor-major, `box int8 (1, 3549, 4)` (left, top, right, bottom distance
from each anchor, in anchor strides) and `cls int8 (1, 3549, 5)` (sigmoid
score of bicycle, car, motorcycle, bus, truck): the NPU's NHWC order
flattened, where the channel-major `(1, C, N)` of Ultralytics cost 209
per-channel copies on the Ethos-U55.

The weights are YOLO26n without its two attention blocks, fine-tuned on the
COCO images with vehicles ([`model/yolo26n-noattn.yaml`](../model/yolo26n-noattn.yaml),
`traffic/train_vehicles.py`, see "Training" below): the whole network is one
method, `detect`, every operator on the NPU. The checkpoint is committed as
[`model/yolo26n-noattn-vehicles.pt`](../model/yolo26n-noattn-vehicles.pt)
(derived from Ultralytics' `yolo26n.pt`, under its AGPL-3.0 license).
Without it, or with `TRAFFIC_WEIGHTS` pointing at the COCO-pretrained
`yolo26n.pt`, the export keeps the attention and moves it to the CPU, as
follows.

The two attention blocks of YOLO26n (the PSA in layer 10 and in layer 22,
13x13 tokens, 2 heads) run on the CPU. The Ethos-U55 has no matrix multiply:
Vela runs each q k^T and each attention-weighted sum as one broadcast
multiply with int32 results plus one REDUCE_SUM per row, 1358 operators, 8
bytes of SRAM traffic per MAC, about 50 of the 80 ms of the whole network
for 1 % of its MACs (see "Performance on the board"). So the export
(`TRAFFIC_ATTENTION=cpu`, the default) is three NPU methods with the
attention cores in between, on the CPU (`traffic/attention.c`):

| Method | Inputs | Outputs |
|--------|--------|---------|
| `stem` | `image int8 (1, 416, 416, 3)`: RGB888 as the camera path delivers it, `q = pixel ^ 0x80` | layers 0-9 and layer 10 up to the attention: `l4`, `l6` (for the neck), `y10` (C2PSA's cv1), `qkv10 (1, 13, 13, 256)` |
| attention core (CPU) | `qkv10` | `o10 (1, 13, 13, 128)` = softmax(q k^T / sqrt(32)) v per head |
| `mid` | `o10`, `qkv10`, `y10`, `l4`, `l6` | the rest of layer 10, the neck, layer 22 up to its attention: `qkv22`, `z22`, `y22`; the stride 8 and 16 heads: `box34`, `cls34` (logits) |
| attention core (CPU) | `qkv22` | `o22` |
| `head` | `o22`, `qkv22`, `z22`, `y22`, `box34`, `cls34` | `box`, `cls` as above |

Every tensor crosses in NHWC, so no method needs a transpose. The same
tensor gets other quantization parameters as an output of one method and an
input of the next (concatenations share one range), so `detector.cpp`
requantizes it in place with a 256-entry table. The stride 32 scores are
appended in `head` so that `cls` keeps one range: a score output of the
stride 8 and 16 anchors alone got the range of what COCO128 scores there (up
to 0.19) and clipped every stronger detection. `TRAFFIC_ATTENTION=npu`
exports the whole network as one method, `detect`, instead (the attention in
the channels-last form of `TokenMajorAttention`, which saves Vela 870
per-channel copies against Ultralytics' form); `detector.cpp` handles both.

Vela 5.1 for the Ethos-U55-256 (`RTSS_HP_SRAM_MRAM`, `Shared_Sram`), every
operator on the NPU: without attention one method, 1.12 GMAC, 1220 KiB SRAM
scratch, a 2.28 MB program; with the attention on the CPU three methods,
1.11 GMAC, 1219 KiB scratch (stem; mid 1156 KiB, head 223 KiB), 2.27 MB.

`traffic/eval_vehicles.py` measures the vehicle AP50 of the float and the
int8 model on a dataset in the YOLO layout; the int8 score output is
calibrated on the COCO128 images with the most vehicles. For the split
export the int8 model is the three quantized methods with the integer
attention core in between (`attention_core_int8()`, what `attention.c`
computes; on the board its output matches the host's to the last bit in all
but 1 of 21632 values).

## Training

`traffic/train_vehicles.py prepare` puts the COCO 2017 train images with a
vehicle (19759) and 5000 without one, with their 80-class labels, into
`tmp/train_vehicles/` (4.2 GB, the images one by one from
images.cocodataset.org); validation is the 570 COCO val2017 images with a
vehicle that are not among the first 300 (those are the test set of
`eval_vehicles.py`). `traffic/train_vehicles.py train` fine-tunes
`model/yolo26n-noattn.yaml` from `yolo26n.pt` (635 of its 720 tensors; the
two replaced blocks start from scratch) at 416 x 416 for 50 epochs with
Ultralytics' defaults (MuSGD, lr 0.01), on an M4 Max's GPU (MPS) in 9 hours,
and copies the best weights to `model/yolo26n-noattn-vehicles.pt`.
The first epochs lose accuracy (the learning rate ramps up while the new
blocks are random), the rest win it back and more:

| Weights | Vehicle AP50 float | int8 |
|---------|--------------------|------|
| COCO-pretrained YOLO26n (attention on the CPU) | 0.476 | 0.464 |
| without attention, fine-tuned, epoch 9 of 50 | 0.395 | 0.374 |
| without attention, fine-tuned, 50 epochs | 0.510 | 0.501 |

(300 COCO val2017 images with vehicles, 1168 of them, `eval_vehicles.py`.)
The fine-tune on vehicles at the application's input size is worth more
than the attention blocks were.

## Build and run

```bash
./setup_venv.sh && .venv/bin/python -m pip install ultralytics   # once
MODEL_FLAVOR=traffic .venv/bin/python create_ai_layer.py ai_layer_traffic/cmsis-executorch.cbuild-mlops.yml
.venv/bin/python traffic/make_test_image.py      # the test image, traffic/test_image.c (not committed)
cbuild cmsis-executorch.csolution.yml --active AppKit-E7 --packs
```

The Secure Enclave has to boot the HP core from its MRAM region: the "Alif:
Install M55_HP debug stubs (AppKit-E7)" task (SW4 on SEUART; the boot table
is `.alif/M55_HP_mram_cfg_e7.json` with the E7's own device configuration
`.alif/app-device-config-e7.json`, silicon revision B4), or
`tools/setools_mram.py --part E7` to program the image itself through the
SE. Then load and debug `AppKit-E7` from the CMSIS view, or the launch
configuration "M55_HP JLink AppKit-E7 traffic".

The console is a log buffer in the DTCM, `console_log`, that the debugger
reads (`JLINK_DEVICE=AE722F80F55D5LS_M55_HP python tools/devkit.py log
out/traffic/AppKit-E7/Release/traffic.axf.map`). The global `traffic_status`
holds the tallies (`counts`), the frame counters, the times of each step in
microseconds, the camera gain and white balance, the live tracks and the
latest detections; writing 1 to `traffic_reset_counts` zeroes the tallies.

## The pipeline

| Step | Where |
|------|-------|
| The kit's MT9M114 module, 640x480 RGB565 from the sensor's own ISP over MIPI CSI-2 (1 lane) into the CPI, which writes one frame buffer; at each VSYNC the application hands it the other of two (`camera.c`; the E7 has no ISP and its CPI no streaming mode) | CPI |
| The frame scaled to 416x416 interleaved RGB888 (an ARX3A0's raw Bayer frame demosaiced first, the colours of each 2x2 cell, no interpolation), turned if the module is mounted turned, gray-world white balance; the MT9M114 runs its own auto exposure, for an ARX3A0 the mean brightness drives the sensor gain (`camera_auto_exposure`) | camera thread (`image.c`) |
| RGB888 to int8 in the backend's input copy, `stem`, `mid` and `head` on the NPU, the two attention cores and the requantization between them on the CPU (`attention.c`), the best class per anchor and the decode | vision thread (`detector.cpp`) |
| Detections matched to the tracks by overlap with a constant-velocity prediction; a track is confirmed after 3 detections and counted once when its centre crosses the line (`tracker.c`) | vision thread |
| The picture into the back frame buffer | vision thread |
| The line, the boxes in the class colours (the track labels kept inside the picture: the margins beside it are never redrawn), the tallies, the score maps, cache clean, present | display thread, while the NPU runs the next frame |

The counting line is vertical in the middle of the picture
(`TRAFFIC_LINE_VERTICAL: 1`, set for the Amsterdam crossing of the YouTube
recordings, whose traffic runs left and right below the middle: over the whole
video a vertical line in the middle counts 261 crossings, a horizontal one
29); without the define it is horizontal. `TRAFFIC_LINE_POS` (input pixels)
and `TRAFFIC_LINE_VERTICAL` in the `define:` block of
`traffic/traffic.cproject.yml` move and turn it, the joystick at run time.
Direction 0 ("DOWN", or "RIGHT" for a vertical line) is a crossing towards
larger coordinates.

Camera settings to confirm on the board (`traffic/camera.h`): `CAMERA_BAYER`
(the Bayer order of the ARX3A0's frame, GRBG assumed: swapped red and blue or
a green cast mean another order) and `CAMERA_QUARTER_TURNS` (the module's
mounting). The pack's own AppKit-E7 layer names the MT9M114 sensor; with that
module the layer's sensor component and `RTE_MT9M114_CAMERA_SENSOR_MIPI_IMAGE_CONFIG 3`
(640x480 RGB565) select the RGB565 path of `camera.c`.

## Recording and playback (SDS)

As for the cat detector: the streams `CameraIn` (each 416x416 RGB888 model
input) and `Detections` (a `detections_t` per frame, now with a class per
box), metadata in `recordings/traffic/*.sds.yml`.

```bash
.venv/bin/python traffic/sds_session.py record 20     # 20 s from the camera
.venv/bin/python traffic/sds_session.py play          # CameraIn.<n>.sds into the detector instead of the camera
.venv/bin/python traffic/images_to_sds.py make recordings/traffic/playback img1.jpg img2.jpg ...
.venv/bin/python traffic/sds_session.py play --workdir recordings/traffic/playback
.venv/bin/python traffic/images_to_sds.py check recordings/traffic/playback   # board vs host float model
```

Playback keeps the recorded pace: every record carries its time in ms (the
kernel tick when recorded from the camera; `images_to_sds.py make --fps N`
stamps images 1000/N ms apart, default 10 fps), and `rec_play_read_input`
holds a record until it is as far after the previous one as their times are
apart. A late record goes at once and the schedule moves with it, so a stall
in the USB transfer does not make the following frames race. Without it
playback ran as fast as the loop, 2.65x too fast for a 10 fps stream at 26.5
fps. Measured over the User USB (2026-09-29): 100 records at 10 fps took
10.0 s from open to close (9.9 s of video); 300 records at 30 fps took 11.6
s, 25.8 fps: the loop cannot keep up with 30, so such a stream plays at
0.86x and never waits. `recordings/traffic/youtube_full_fill` is the traffic
video (25 fps, 9 min) resampled to 30 fps by ffmpeg, which repeats every
sixth frame: 16303 records, 8.5 GB; `recordings/traffic/youtube_60s` is its
first minute (1800 records, 935 MB), which plays in 67 s (26.8 fps; with the
vertical line `count_playback.py` counts 26 over the board's detections:
22 bicycles, 2 cars, 2 motorcycles).
`images_to_sds.py` at 10 fps (eval) makes
300 images a 30 s playback. The tracker counts per frame, not per second:
the pace changes what the panel shows, not the counts.

On the AppKit-E7 the joystick moves the line (`traffic/joystick.c`,
`APP_HAS_JOYSTICK`): left and right move a vertical line, up and down a
horizontal one, faster once held for a second, and the centre button turns it
by 90 degrees; the direction labels follow. The switches sit on the
low-power GPIO block (P15_0..4, pulled up by the pin table); a read of that
block costs milliseconds, so a thread at the lowest priority polls it every
40 ms while the vision thread waits for the NPU (from the vision thread the
five reads took 17 ms a frame and the panel flickered), and the vision thread
applies the requested line between two tracker updates; moving the line
counts nothing by itself.

`traffic/count_playback.py <dir> [--sweep]` reruns tracker.c's logic on the
PC over the board's detections of a playback (the count the board showed) and
over the float model's detections of the same inputs (the reference), by
class and direction, and reports where the tracks moved; `--sweep` tries
horizontal and vertical lines at several positions. On the YouTube clip
(vehicles along a road across the picture just below the middle) the
horizontal line at 208 counts 9 on the board and 12 on the host, a vertical
line at 208 counts 45 and 64: the line has to cut the traffic's path
(`TRAFFIC_LINE_POS`, `TRAFFIC_LINE_VERTICAL` in the layer).

The firmware's SDSIO client connects only while it starts: start the server
(`sds_session.py`, which waits for the client's first flags exchange), then
reset the board. `traffic/eval_vehicles.py <dataset> --board <playback
folder>` scores the board's detections of a playback of the dataset's images
against the host's, all cut as `detector.cpp` cuts them (score above 0.30, at
most 16 boxes). For the 300 test images (`recordings/traffic/eval300`,
2026-09-29, the fine-tuned model without attention) the board matches the
host's int8 model: AP50 of the cut detections 0.362 (host int8 0.362, float
0.366), precision 0.809 (0.808), recall 0.396 (0.397), NPU 31.1 ms per image.

The same four steps are VS Code tasks (Terminal > Run Task): "SDS: video to
CameraIn stream" (ffmpeg frames at a chosen rate, letterboxed or with
`--fill`, the sides cut so the picture fills the input), "SDS: play recording
to the board (USB)", "SDS: record from the camera (USB)" and "SDS: check
playback against the host model"; and the panel video below, "SDS: play
recording and record the panel (USB)" and "SDS: panel stream to MP4". One
SDSIO server owns the USB device at a time: a second playback started while
one runs resets the link, and the board's half-open streams then crash the
next server; let a run finish or stop it with Ctrl+C.

### A video of the panel

With SDSIO-Server's user flag 0 set when the streams open (key A, or
`sds_session.py --panel`) the firmware opens a third stream, `Panel`: every
frame as the panel showed it, the CDC200's whole 480x800 frame buffer. The
display thread writes it after the frame is on the panel and before the
vision thread may draw into that buffer again, so no frame is lost and a slow
link holds the loop back instead. A record must fit the stream buffer and no
memory holds a 1.15 MB one, so a frame goes as 20 records of 40 rows with the
frame's timeslot, through a 230 kB buffer in SRAM1 (four records); a lock
keeps the vision thread from closing the stream in the middle of a frame,
and at the end of a playback it waits for the last frame's records.
`recordings/traffic/Panel.sds.yml` describes a record; the bytes are the
frame buffer's, B, G, R per pixel and turned by 180 degrees.

```bash
.venv/bin/python traffic/sds_session.py play --panel --timeout 3600 --workdir recordings/traffic/<folder>
.venv/bin/python traffic/panel_to_video.py recordings/traffic/<folder> -o panel.mp4
.venv/bin/python traffic/panel_to_video.py recordings/traffic/<folder> --still 150 -o frame.png
```

`panel_to_video.py` joins the records of each frame, turns it upright and
has ffmpeg write H.264 at the pace of the timeslots: in a playback those are
the clip's (30 fps for the traffic video), so the video shows the traffic at
its real speed, every frame the board drew, whatever rate the board managed
while it streamed; `--fps` sets another. A frame cut short (stopped with S
while it was written) is dropped. SDS-Convert's `video` mode cannot do this:
it reads the whole file into memory and takes one record per frame.

The cost is the USB: 1.15 MB up per frame next to the 519 kB CameraIn record
down. On the whole traffic video (`youtube_full_fill`, 16303 frames,
2026-10-01) the loop ran at 17.1 fps instead of 25.8, 58 ms a frame of which
24 ms the vision thread waited for the display thread's panel write; the
NPU's 31 ms did not change. The playback took 16 minutes, about 29 MB/s over
the User USB, and every frame arrived: 16303 panel frames (18.8 GB) for
16303 detections. The "FPS" on the recorded panel is that slower rate.

During a playback the camera thread pauses: the model inputs share the two
slots in SRAM1 (see below). It also starts no snapshots then: while it kept
the CPI going unread, playbacks of the traffic video stopped after 580 and
1420 frames in `osRtxErrorNotify` (stack overflow of `sdsControl`), a
snapshot had written about 1 kB of picture past the end of the last camera
buffer, over the bottom of that thread's stack right behind it (3 CPI error
events counted by then).

The link is the board's User USB (`SDS:IO:USB&MDK USB`, a custom-class
device "SDSIO-Client", VID 0xC251 PID 0x8007, high speed): the 40-frame
YouTube clip (20.7 MB) plays in about 8 s, where the J-Link RTT link
(`--transport rtt`, `tools/sdsio_rtt_bridge.py`) needed 5 s per frame.
Two things the USB device needs on the E7: the application powers the USB
PHY through the Secure Enclave (`usb_power_init` in `main.c`), and every
buffer the USB controller reads or writes in the DTCM must lie in the
linker's non-secure region `NS_REGION_0` (the TGU rejects the controller
elsewhere, and a control or bulk transfer then hangs silently): the EP0
buffer (`USBD0_BUF_MEM_LOCATE`), the SDS client's bulk buffer, the SDS
data block, and the RTX dynamic memory that holds the SDS thread's stack,
because the SDS client sends its command headers straight from the stack.

## Memory

| Memory | Holds |
|--------|-------|
| MRAM, from 0x80200000 | code and the test image in the HP core's region, then the 2.46 MB program (`ER_MODEL`, running on into the MRAM user region at 0x80400000). It stops 64 kB below the top of the MRAM, where the Secure Enclave's application package lives (`board/AppKit-E7/linker_ac6_traffic.sct.src`); 3.0 of the 3.4 MB are used |
| SRAM0, 4 MB | the ExecuTorch method pool, the two 640x480 RGB565 camera frames (614 kB each), the two 480x800 RGB888 panel buffers (1.15 MB each), the thread stacks; the first 4 kB stay free for the Cortex-A32 stub; 92 % used (with the attention on the CPU also the outputs of `stem`, 519 kB, reused by `head`'s, with a 16 kB method pool: 99 %) |
| SRAM1, 2.5 MB at 0x08000000 | the `Panel` stream buffer (230 kB), two model input slots (519 kB each) and Vela's scratch (1.25 MB, `APP_TEMP_POOL_SECTION`; the NPU runs 81 ms from here as from SRAM0); 98 % used; `UNINIT`, powered through the Secure Enclave by the application; the CPI and the CDC200 cannot reach it, so the camera frames and the panel buffers are in SRAM0. SRAM8 (2 MB at 0x63200000) is no alternative for a panel buffer: with it the display DMA hangs the interconnect at boot, the debug port with it, and the image in MRAM then has to be replaced with the reset button pressed while the J-Link connects |
| DTCM, 1 MB | the console log, the SDS input buffer, the C library, the tracker (with the attention on the CPU also the outputs of `mid`, 139 kB, the two attention outputs and the attention core's buffers: 92 % instead of 72 %); the non-secure region at its end holds the USB DMA buffers and RTX's dynamic memory |

Unlike the E8's, the E7's SRAM0 and SRAM1 are not contiguous, so the scatter
file places them separately; the input slots went from three to two to fit.

## Camera on the AppKit-E7

Four things stood between the pack's AppKit-E7 BSP and a live picture
(2026-09-25, all found on the board with the debugger):

- **The camera clock pin.** The BSP's pin table selects `LPCAM_XVCLK_B` on
  P0_3, the clock of the HE core's low-power CPI, which nothing enables. The
  sensor drivers run the HP CPI's pixel clock (`CAMERA_PIXCLK_CTRL`, 400 MHz /
  20), which reaches the pin only as `CAM_XVCLK_A`: without it the sensor never
  answers on I2C1. `board/AppKit-E7/RTE/BSP/.../pins.h` selects `CAM_XVCLK_A`,
  as the pack's own AppKit-E7 layer does.
- **The sensor.** The kit's module is an MT9M114 (I2C address 0x5D over MIPI),
  not an ARX3A0: the layer names the MT9M114 component,
  `RTE_MT9M114_CAMERA_SENSOR_MIPI_IMAGE_CONFIG 3` (640x480 RGB565). Its ISP
  tracks the exposure only once asked (`CPI_CAMERA_SENSOR_AE` in `camera.c`).
- **SRAM1 is the CPU's alone.** The CPI's writes to SRAM1 come back with AXI
  decode errors (`CAM_AXI_ERR_STAT` 0xff03), and the CDC200 cannot read it
  either (the panel goes dark). Neither the E8's nor the E7's device
  configuration in the boot table changes that. So the camera frames live in
  SRAM0 (`.bss.ai_pool`); to keep both panel buffers there too, Vela's scratch
  moved to SRAM1 next to the model input slots (see Memory).
- **One snapshot per frame.** In video mode the pack's driver refuses a new
  frame address while the CPI is busy (`ARM_DRIVER_ERROR_BUSY`), and the E7's
  CPI, which has no stream mode, then writes frame after frame past the
  buffer, over the thread stacks and the pools: a bus fault in the camera
  thread, and within seconds the Secure Enclave and the debug port are gone.
  `camera.c` takes one snapshot at a time (`CaptureFrame`, the STOP event) and
  starts the next one into the other buffer from the camera thread, about 25
  snapshots a second.

The panel's MADCTL value is a layer setting (`TRAFFIC_LCD_MADCTL`), written
before the video stream starts. On this ILI9806E in video mode only bit 0
does anything, a mirror along the long axis (the pack's panel driver sets
0x01, the layer 0x00); bits 7 and 6 are ignored. The panel hangs upside down, so
`IMAGE_PANEL_TURN_180` (the layer) makes `image.c` draw everything, the picture
and every rectangle behind boxes, line, score maps and text, at the opposite
x and y: no extra pass over the frame buffer (turning it in place afterwards
cost 60 ms a frame and starved the NPU).

The panel buffers are the CDC200's RGB888 (`RTE_CDC200_PIXEL_FORMAT 1`): the
HWRM names the layer formats by bit position in a little-endian word
(`CDC_Ln_PIX_FORMAT`: ARGB8888, RGB888, RGB565 with red in the top bits), so
RGB888 is R in bits 23:16 and B in 7:0, in memory B, G, R. The model input
is R, G, B, which the MT9M114's RGB565 (R in bits 15:11) gives directly and
the model needs, so `image.c` swaps R and B when it copies the picture into
the panel buffer and when it fills a rectangle. With R, G, B in the panel
buffer red and blue trade places on the panel.

With the camera the loop runs at 25 fps (18.7 fps with the attention on the
CPU). The camera thread converts a
frame only when the vision thread has taken the last one (the camera
delivers about twice the frames the detector takes), and converts below the
vision thread's priority, so the one conversion per frame (7 ms, 15 ms before
the per-pixel division left `image_rgb565_to_input`) goes into the vision
thread's waits for the NPU and not into the detector's CPU work. Right after a
`load_and_debug` (a core-only reset) the first inference can fail with err 35,
the NPU still busy with the previous job; the loop recovers on its own.

## Performance on the board

2026-09-28/29, test image, no camera:

| Model and export | NPU | CPU attention | Requant | Detect | Frame | Camera loop | Test image (host int8) |
|------------------|-----|---------------|---------|--------|-------|-------------|------------------------|
| YOLO26n, `detect`, everything on the NPU (before) | 80.4 ms | - | - | 82.2 ms | 85.6 ms, 11.7 fps | 11.1 fps | truck 0.69, truck 0.61 |
| YOLO26n, `detect`, channels-last attention, anchor-major outputs | 79.7 ms | - | - | | | | truck 0.69, truck 0.61 |
| YOLO26n, `stem` + `mid` + `head`, attention on the CPU | 31.1 ms (16.4 + 13.0 + 1.6) | 10.0 ms | 1.7 ms | 47.3 ms | 50.7 ms, 19.7 fps | 18.7 fps | truck 0.69, truck 0.61 (0.69, 0.61) |
| without attention, fine-tuned, `detect` | 31.0 ms | - | - | 32.7 ms | 36.0 ms, 27.6 fps | 25.0 fps | truck 0.65, truck 0.50 (0.65, 0.50, and a motorcycle at 0.35) |

With the attention on the CPU the NPU writes 12.5 MB per frame instead of 65
MB; the rest of its `detect` is the backend's copies in and out of the three
NPU jobs (about 4.5 ms). The vehicle AP50 on 300 COCO val2017 images with
vehicles (1168 of them) of YOLO26n: float 0.476; int8 0.457 before, 0.461
with the channels-last attention, 0.464 with the attention on the CPU; a
smaller input costs accuracy (int8 0.420 at 384, 0.403 at 352, 0.386 at
320). Without attention and fine-tuned: float 0.510, int8 0.501.

What the attention costs on the NPU was found by running the command stream
from a copy in SRAM0 with an NPU_OP_STOP (and QSIZE right after it) at the
first attention operator: the backbone before it takes 16.5 of the 80.4 ms,
22.0 of the 102 MB read and 6.3 of the 65 MB written.

Before (2026-09-25), AppKit-E7, HP core and Ethos-U55-256 at 400 MHz, the
test image, no camera. The NPU time is the command stream from start to interrupt,
the rest of the frame is the CPU (the input copy, the decode, the tracker, the
picture); the display thread runs while the NPU works.

| Input | Vela | NPU | Frame | Detections on the test image |
|-------|------|-----|-------|------------------------------|
| 416 | `RTSS_HP_SRAM_MRAM`, `Shared_Sram` | 81.0 ms | 83.4 ms, 12.0 fps | 2 (float model on the host: 0.71, 0.56, 0.35, 0.27) |
| 416 | `RTSS_HP_DTCM_SRAM`, `Dtcm_Cache`, 640 kB cache in the DTCM | 84.0 ms | 86.4 ms, 11.6 fps | 2 |
| 320 | `RTSS_HP_SRAM_MRAM`, `Shared_Sram` | 40.6 ms | 42.2 ms, 23.7 fps | 0: the int8 model's best score is 0.28, below the 0.30 threshold (host: 0.47, 0.35, 0.28) |

The U55 is bound by its one AXI port, not by the MACs and not by the MRAM.
Its PMU over one 416 inference (32.4 M cycles, counters programmed from the
debugger after the driver's soft reset, `detector.cpp`'s inference hooks):

| Counter | Cycles or beats | Share |
|---------|-----------------|-------|
| `MAC_ACTIVE` | 7.6 M | 23 % |
| `MAC_STALLED_BY_IB` (waiting for input data) | 10.7 M | 33 % |
| `AXI0_RD_TRAN_REQ_STALLED` | 11.9 M | 37 % |
| `MAC_STALLED_BY_WD` (weights from the MRAM) | 1.1 M | 3 % |
| `AXI0_RD_DATA_BEAT_RECEIVED` | 12.8 M beats in 3.1 M transactions (4 beats each) | 102 MB read |
| `AXI0_WR_DATA_BEAT_WRITTEN` | 8.1 M beats | 65 MB written |

The same counters with the scratch cache in the DTCM are unchanged (the DTCM
is no faster for the NPU than SRAM0 on this part, whatever `ensemble_vela.ini`
assumes), so memory placement does not help; only less traffic does. A 320
input halves the NPU time but costs detections: the model would need the
score threshold and the int8 calibration revisited at that size
(`YOLO_IMGSZ=320` for `create_ai_layer.py` and `make_test_image.py`). Most of
the traffic was the attention (above).

## Status

Runs on the AppKit-E7 (AK-E7-AIML, Gen 2, MT9M114 module; 2026-09-28, AC6,
`out/traffic/AppKit-E7/Release/traffic.axf`): YOLO26n without attention,
fine-tuned on vehicles, the camera loop at 25 fps (11.5 fps with the
COCO-pretrained YOLO26n whole on the NPU, 18.7 fps with its attention on the
CPU), the NPU at 31 ms per 416x416 frame, the joystick line, and SDS
recording and playback over the User USB. The tracker's counting is checked
on the host against the float model (`traffic/count_playback.py`).
