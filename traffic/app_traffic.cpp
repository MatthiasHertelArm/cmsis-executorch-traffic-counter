/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Traffic counter with YOLO26n on the Ethos-U55 of the Alif Ensemble E7
 * (M55-HP, AppKit-E7), live from the camera to the panel, with SDS recording
 * and playback.
 *
 * Each frame:
 *   1. the input: the newest camera frame (the ARX3A0's RAW8 Bayer, or an
 *      MT9M114's RGB565), demosaiced, white-balanced and scaled to the
 *      416x416 RGB888 model input by the camera thread; or, in playback, the
 *      next CameraIn record from SDSIO-Server; without a camera, the test
 *      image (traffic/test_image.c)
 *   2. recording: the input goes into the CameraIn stream
 *   3. the detector: YOLO26n on the NPU, the head's decode on the CPU (detector.cpp)
 *   4. the tracker: the detections matched to the tracks, the line crossings counted (tracker.c)
 *   5. the result goes into the Detections stream while recording or playing back
 *   6. the panel: the picture in a 480x480 view; the tracks, the line, the
 *      tallies and the score maps around it in the display thread, while the
 *      NPU works on the next frame; double-buffered. With SDSIO-Server's flag
 *      0 set the shown frame goes into the Panel stream too.
 *
 * The console is a log buffer that the debugger reads (board layer); the
 * counts, the latest result and the frame timing are in the global
 * `traffic_status`; `traffic_reset_counts` set to 1 by the debugger zeroes
 * the tallies.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "cmsis_os2.h"
#include "se_services_port.h"
#include "services_lib_api.h"

#include "detector.h"
#include "image.h"
#ifdef TRAFFIC_LCD_MADCTL
extern "C" {
#include "DSI_DCS.h"  // the pack header has no C++ guard
}
#endif
#include "model_pte.h"
#include "test_image.h"
#include "tracker.h"
#ifdef APP_HAS_DISPLAY
#include "board_display.h"
#endif
#ifdef APP_HAS_CAMERA
#include "camera.h"
#endif
#ifdef APP_HAS_SDS
#include "rec_play.h"
#endif
#ifdef APP_HAS_JOYSTICK
#include "joystick.h"
#endif

#ifndef TRAFFIC_LINE_POS
#define TRAFFIC_LINE_POS (TEST_IMAGE_SIZE / 2)
#endif
#ifndef TRAFFIC_LINE_VERTICAL
#define TRAFFIC_LINE_VERTICAL 0
#endif

// ---------------------------------------------------------------------------
// What the debugger reads.
// ---------------------------------------------------------------------------
struct TrafficStatus {
  uint32_t frames;          // frames processed
  int32_t source;           // 0 test image, 1 camera, 2 SDS playback
  int32_t mode;             // rec_play_mode_t
  int32_t camera_status;    // camera_init(): 0 ok, else the failing step; -1 no camera support
  uint32_t camera_frames;   // frames the camera delivered
  uint32_t camera_errors;
  uint32_t camera_gain;     // the sensor gain the auto exposure set, 16.16
  uint32_t camera_mean;     // mean raw green of the last frame
  int32_t detector_status;  // last ExecuTorch error, 0 ok
  uint32_t frame_us;        // last frame, input to presented
  uint32_t input_us;        // waiting for the camera input or the playback read
  uint32_t convert_us;      // camera thread: raw frame to model input
  uint32_t detect_us;       // detector_run: execute and decode
  uint32_t npu_us;          // the NPU job alone
  uint32_t track_us;        // tracker_update
  uint32_t picture_us;      // the input into the frame buffer (vision thread)
  uint32_t display_us;      // display thread: boxes, text, heat maps and cache clean
  uint32_t panel_us;        // display thread: the shown frame into the SDS Panel stream
  uint32_t sds_us;          // SDS writes
  float fps;                // frames per second, averaged over one second
  float wb_gain[3];         // white balance gains R, G, B
  uint32_t tracks;          // live tracks
  uint32_t tracks_confirmed;
  traffic_counts_t counts;  // the tallies
  detections_t result;      // the detections of the last frame
};

extern "C" {
volatile TrafficStatus traffic_status;
volatile uint32_t traffic_reset_counts;  // set to 1 to zero the tallies
// The benchmark build's result (TRAFFIC_BENCHMARK).
struct {
  int32_t status;
  uint32_t runs, count;
  uint32_t total_us_min, total_us_sum, npu_us_min, npu_us_sum;
} volatile traffic_benchmark;
// Every 8th pixel of the last model input (52 x 52 RGB888): small enough for
// a debugger to read in two requests and look at the picture on the host.
uint8_t traffic_thumbnail[52 * 52 * 3];
}

namespace {

constexpr int kSize = TEST_IMAGE_SIZE;
constexpr uint32_t kInputBytes = kSize * kSize * 3;
constexpr int kSlots = 2;

// The model inputs: two slots in SRAM1 that the camera thread fills in turn;
// without a camera (the test image, playback) slot 0 is the input.
alignas(32) uint8_t g_slot[kSlots][kInputBytes] __attribute__((section(".bss.sram1")));

#ifdef APP_HAS_DISPLAY
// Two frame buffers: the picture and the overlays go into the one the panel
// does not show, then the panel switches. With one buffer the picture copy
// wiped the overlays a few milliseconds before they were drawn again, and
// boxes, line and text flickered. On the E7 the second buffer fits SRAM0 only
// because the NPU's scratch went to SRAM1 (detector.cpp, the layer's
// APP_TEMP_POOL_SECTION); the CDC200 cannot read SRAM1, and SRAM8 hangs the
// bus (see the documentation).
constexpr uint32_t kFrameBytes = IMAGE_PANEL_W * IMAGE_PANEL_H * 3;
alignas(32) uint8_t g_framebuffer0[kFrameBytes] __attribute__((section(APP_FRAMEBUFFER_SECTION)));
alignas(32) uint8_t g_framebuffer1[kFrameBytes] __attribute__((section(APP_FRAMEBUFFER_SECTION)));
uint8_t* const g_framebuffer[2] = {g_framebuffer0, g_framebuffer1};
#endif

// SRAM1 (2.5 MB at 0x08000000) holds the camera frames, the input slots and
// the SDS input buffer. Power and clock it through the Secure Enclave before
// any use; a store to an unpowered SRAM hangs the bus.
bool sram1_power_on() {
  uint32_t error = 0;
  if (SERVICES_power_memory_req(se_services_s_handle, POWER_MEM_SRAM_0_ENABLE | POWER_MEM_SRAM_1_ENABLE, &error) !=
          SERVICES_REQ_SUCCESS ||
      error != 0) {
    printf("SRAM1: power request failed (error %lu)\n", static_cast<unsigned long>(error));
    return false;
  }
  if (SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_SRAM1, true, &error) != SERVICES_REQ_SUCCESS || error != 0) {
    printf("SRAM1: clock request failed (error %lu)\n", static_cast<unsigned long>(error));
    return false;
  }
  return true;
}

inline uint32_t cycles() { return DWT->CYCCNT; }
inline uint32_t us(uint32_t c) { return static_cast<uint32_t>(static_cast<uint64_t>(c) * 1000000U / SystemCoreClock); }

enum Source { kTestImage = 0, kCamera = 1, kPlayback = 2 };

#ifdef APP_HAS_CAMERA
// The camera thread: every camera frame turned into a model input in the slot
// the vision thread is not reading. The vision thread takes the newest slot
// and holds it for the frame; the camera thread writes the other one, again
// and again, so the vision thread always finds the newest complete frame.
// A camera frame that arrives while a converted one still waits unread is
// skipped: the camera delivers about twice the frames the detector takes, and
// a conversion (7 ms) that nobody reads costs the detector CPU time. The
// thread takes the frames above the vision thread (the next snapshot starts
// in camera_frame()) but converts below it, so the one conversion per frame
// goes into the vision thread's waits for the NPU, not into the detector's CPU
// work (the attention cores and the copies between its NPU methods).
volatile int g_slot_newest = -1;   // the newest complete input, -1 when taken
volatile int g_slot_reading = -1;  // the vision thread's input
volatile bool g_camera_pause;      // playback: the slots belong to the vision thread
volatile bool g_camera_busy;       // the camera thread is writing a slot
osEventFlagsId_t g_slot_event;
uint64_t g_camera_stack[512] __attribute__((section(APP_POOL_SECTION)));
volatile uint32_t g_convert_cycles, g_camera_mean;

__NO_RETURN void camera_thread(void*) {
  for (;;) {
    // No snapshots during a playback: nobody reads them, and during playbacks
    // a snapshot wrote about 1 kB past the end of the last camera buffer,
    // into the SDS control thread's stack behind it (RTX then stopped in
    // osRtxErrorNotify, stack overflow, after 580 and 1420 frames).
    // camera_frame() starts the next snapshot when the playback is over.
    if (g_camera_pause) {
      osDelay(10);
      continue;
    }
    const void* frame = camera_frame(200);
    if (frame == nullptr || g_camera_pause || g_slot_newest >= 0) continue;
    osKernelLock();
    int slot = 0;
    while (slot == g_slot_reading) ++slot;
    if (g_slot_newest == slot) g_slot_newest = -1;  // being rewritten: not to be taken meanwhile
    g_camera_busy = true;
    osKernelUnlock();
    osThreadSetPriority(osThreadGetId(), osPriorityBelowNormal);
    const uint32_t t0 = DWT->CYCCNT;
#if CAMERA_RAW8
    uint32_t mean = 0;
    image_bayer_to_input(static_cast<const uint8_t*>(frame), CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_BAYER, g_slot[slot], kSize,
                         CAMERA_QUARTER_TURNS, &mean);
    g_camera_mean = mean;
    camera_auto_exposure(mean);
#else
    image_rgb565_to_input(static_cast<const uint16_t*>(frame), CAMERA_WIDTH, CAMERA_HEIGHT, g_slot[slot], kSize,
                          CAMERA_QUARTER_TURNS);
#endif
    g_convert_cycles = DWT->CYCCNT - t0;
    osThreadSetPriority(osThreadGetId(), osPriorityAboveNormal);
    osKernelLock();
    g_slot_newest = slot;
    g_camera_busy = false;
    osKernelUnlock();
    osEventFlagsSet(g_slot_event, 1U);
  }
}

// The newest camera input, waiting up to `timeout_ms`; nullptr on a timeout.
const uint8_t* camera_input(uint32_t timeout_ms) {
  for (;;) {
    osKernelLock();
    const int slot = g_slot_newest;
    if (slot >= 0) {
      g_slot_reading = slot;
      g_slot_newest = -1;
    }
    osKernelUnlock();
    if (slot >= 0) return g_slot[slot];
    if (osEventFlagsWait(g_slot_event, 1U, osFlagsWaitAny, timeout_ms) & osFlagsError) return nullptr;
  }
}

void camera_input_done() { g_slot_reading = -1; }

// Playback takes slot 0: the camera thread stops writing until it is over.
void camera_pause(bool pause) {
  g_camera_pause = pause;
  if (pause) {
    while (g_camera_busy) osDelay(1);
    g_slot_newest = -1;
  }
}
#endif

#ifdef APP_HAS_DISPLAY
// Around the picture (480 x 160 above, 480 x 160 below):
//   above: the total, then a row per class with its count and the counts per direction
//   below: the best class score of every anchor as three heat maps (stride 8,
//          16, 32), the anchors above the threshold, the highest score, the
//          timing, the tracks and the input source
void draw_status(uint8_t* fb, const traffic_counts_t& counts, const detector_scores_t* scores, uint32_t tracks,
                 uint32_t confirmed, Source source, int mode, float fps, uint32_t npu_us) {
  constexpr uint32_t kGrey = 0xC0C0C0U, kDim = 0x707070U, kGreen = 0x00FF00U, kRed = 0xFF3030U, kBlue = 0x40A0FFU,
                     kWhite = 0xFFFFFFU;
  image_fill(fb, 0, 0, IMAGE_PANEL_W, IMAGE_VIEW_TOP, 0x000000U);
  image_fill(fb, 0, IMAGE_VIEW_TOP + IMAGE_VIEW, IMAGE_PANEL_W, IMAGE_PANEL_H, 0x000000U);

  char line[48];
  image_text(fb, 12, 6, 2, "TRAFFIC COUNTER  YOLO26N  ETHOS-U55", kDim);
  snprintf(line, sizeof(line), "%lu", static_cast<unsigned long>(counts.total));
  image_text(fb, 12, 24, 5, line, kWhite);
  const char* dir0 = tracker_line_vertical() ? "RIGHT" : "DOWN";
  const char* dir1 = tracker_line_vertical() ? "LEFT" : "UP";
  snprintf(line, sizeof(line), "%s %lu", dir0, static_cast<unsigned long>(counts.by_direction[0]));
  image_text(fb, 160, 26, 2, line, kGrey);
  snprintf(line, sizeof(line), "%s %lu", dir1, static_cast<unsigned long>(counts.by_direction[1]));
  image_text(fb, 160, 46, 2, line, kGrey);
  // One row per class: the count, and per direction.
  for (int c = 0; c < DETECTOR_CLASSES; ++c) {
    const int y = 66 + c * 18;
    const uint32_t colour = counts.by_class[c] ? image_class_colour(c) : kDim;
    image_fill(fb, 12, y + 2, 22, y + 12, image_class_colour(c));
    snprintf(line, sizeof(line), "%-10s %4lu   %s %lu  %s %lu", DETECTOR_CLASS_NAMES[c],
             static_cast<unsigned long>(counts.by_class[c]), dir0, static_cast<unsigned long>(counts.by_class_direction[c][0]),
             dir1, static_cast<unsigned long>(counts.by_class_direction[c][1]));
    image_text(fb, 30, y, 2, line, colour);
  }

  const int y = IMAGE_VIEW_TOP + IMAGE_VIEW + 8;
  if (scores != nullptr) {
    // Three maps of 104 x 104 (the 52 x 52 cells of stride 8 at 2 pixels a cell).
    image_score_maps(fb, 12, y, scores, DETECTOR_INPUT_SIZE, 2);
    image_text(fb, 12, y + 110, 1, "BEST CLASS SCORE PER ANCHOR  S8  S16  S32", kDim);
    snprintf(line, sizeof(line), "MAX %.2f", static_cast<double>(scores->max));
    image_text(fb, 360, y, 2, line, scores->max >= scores->threshold ? kWhite : kGrey);
    snprintf(line, sizeof(line), "HITS %lu", static_cast<unsigned long>(scores->candidates));
    image_text(fb, 360, y + 22, 2, line, kGrey);
    image_fill(fb, 12, y + 128, 26, y + 142, 0xFFFFFFU);
    snprintf(line, sizeof(line), "AT OR ABOVE THRESHOLD %.2f", static_cast<double>(scores->threshold));
    image_text(fb, 34, y + 128, 2, line, kDim);
  }
  snprintf(line, sizeof(line), "%.1f FPS", static_cast<double>(fps));
  image_text(fb, 360, y + 44, 2, line, kGrey);
  snprintf(line, sizeof(line), "NPU %.1f", npu_us / 1000.0);
  image_text(fb, 360, y + 66, 2, line, kGrey);
  snprintf(line, sizeof(line), "TRK %lu/%lu", static_cast<unsigned long>(confirmed), static_cast<unsigned long>(tracks));
  image_text(fb, 360, y + 88, 2, line, kGrey);

  const char* what = source == kCamera ? "CAMERA" : source == kPlayback ? "SDS PLAY" : "TEST IMG";
  uint32_t colour = source == kPlayback ? kBlue : kGreen;
#ifdef APP_HAS_SDS
  if (mode == REC_PLAY_RECORD) {
    what = "REC";
    colour = kRed;
  }
#else
  (void)mode;
#endif
  image_fill(fb, 360, y + 112, 376, y + 128, colour);
  image_text(fb, 382, y + 114, 2, what, colour);
}

// The display thread draws everything but the picture while the vision
// thread runs the NPU on the next frame: the picture goes into the back
// buffer in the vision thread, the rest here, from a copy of the frame's results.
constexpr int kAnchors = (kSize / 8) * (kSize / 8) + (kSize / 16) * (kSize / 16) + (kSize / 32) * (kSize / 32);
struct DisplayJob {
  uint8_t* fb;
  detections_t det;
  track_t tracks[TRACKER_MAX_TRACKS];
  traffic_counts_t counts;
  uint32_t live, confirmed;
  detector_scores_t scores;
  bool has_scores;
  Source source;
  int mode;
  uint32_t timeslot;
  float fps;
};
DisplayJob g_job;
int8_t g_job_scores[kAnchors];
osSemaphoreId_t g_display_go;    // a job is ready
osSemaphoreId_t g_display_idle;  // the last job is on the panel: the other buffer is free
uint64_t g_display_stack[1024] __attribute__((section(APP_POOL_SECTION)));

__NO_RETURN void display_thread(void*) {
  for (;;) {
    osSemaphoreAcquire(g_display_go, osWaitForever);
    const uint32_t t0 = cycles();
    uint8_t* fb = g_job.fb;
    image_draw_line(fb, kSize, tracker_line_pos(), tracker_line_vertical(), 0xFFFFFFU);
    image_draw_detections(fb, &g_job.det, kSize);
    image_draw_tracks(fb, g_job.tracks, kSize);
    draw_status(fb, g_job.counts, g_job.has_scores ? &g_job.scores : nullptr, g_job.live, g_job.confirmed, g_job.source,
                g_job.mode, g_job.fps, g_job.det.npu_us);
    SCB_CleanDCache_by_Addr(fb, static_cast<int32_t>(kFrameBytes));
    traffic_status.display_us = us(cycles() - t0);
    display_present(fb);
    // Presented takes effect at the panel's next refresh: until then the
    // other buffer is still on the panel and must not be drawn over.
    display_wait_shown(fb);
#ifdef APP_HAS_SDS
    // The frame as the panel shows it into the Panel stream (open only when
    // SDSIO-Server's flag 0 asked for it), before the vision thread may draw
    // into this buffer again: a slow link holds the loop back, no frame is lost.
    if (g_job.mode != REC_PLAY_IDLE) {
      const uint32_t t1 = cycles();
      rec_play_write_panel(fb, IMAGE_PANEL_H, IMAGE_PANEL_W * 3, g_job.timeslot);
      traffic_status.panel_us = us(cycles() - t1);
    }
#endif
    osSemaphoreRelease(g_display_idle);
  }
}
#endif

#ifdef APP_HAS_JOYSTICK
// ---------------------------------------------------------------------------
// The joystick: polled from its own thread, because a read of the low-power
// GPIO block takes milliseconds (17 ms for the five switches from the vision
// thread, which made the frame miss the panel's refresh and flicker). At the
// lowest priority the polls run while the vision thread waits for the NPU.
// Left/right move a vertical line, up/down a horizontal one, faster once
// held for a second; the centre turns the line by 90 degrees. The vision
// thread applies the request between two tracker updates.
// ---------------------------------------------------------------------------
volatile int g_line_req_pos = TRAFFIC_LINE_POS;
volatile int g_line_req_vertical = TRAFFIC_LINE_VERTICAL;
uint64_t g_joystick_stack[256] __attribute__((section(APP_POOL_SECTION)));

__NO_RETURN void joystick_thread(void*) {
  constexpr uint32_t kPollMs = 40;
  uint32_t held_polls = 0;
  for (;;) {
    osDelay(kPollMs);
    const uint32_t pressed = joystick_pressed();
    const uint32_t held = joystick_held();
    int pos = g_line_req_pos;
    int vertical = g_line_req_vertical;
    if (pressed & JOY_SELECT) vertical = !vertical;
    const uint32_t back = vertical ? JOY_LEFT : JOY_UP;
    const uint32_t forth = vertical ? JOY_RIGHT : JOY_DOWN;
    held_polls = (held & (back | forth)) ? held_polls + 1 : 0;
    const int step = held_polls > 1000 / kPollMs ? 6 : 2;
    if (held & back) pos -= step;
    if (held & forth) pos += step;
    pos = pos < 0 ? 0 : (pos >= kSize ? kSize - 1 : pos);
    if (pressed & JOY_SELECT) printf("line: %s at %d\n", vertical ? "vertical" : "horizontal", pos);
    g_line_req_pos = pos;
    g_line_req_vertical = vertical;
  }
}
#endif

}  // namespace

#ifndef TRAFFIC_CAMERA_AUTOSTART
#define TRAFFIC_CAMERA_AUTOSTART 1
#endif
extern "C" volatile int traffic_camera_enable = TRAFFIC_CAMERA_AUTOSTART;

extern "C" int app_main(void) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  memset(const_cast<TrafficStatus*>(&traffic_status), 0, sizeof(traffic_status));
  printf("Traffic counter: YOLO26n on the Ethos-U55, input %dx%d RGB888, core %lu MHz, line at %d %s\n", kSize, kSize,
         static_cast<unsigned long>(SystemCoreClock / 1000000U), TRAFFIC_LINE_POS,
         TRAFFIC_LINE_VERTICAL ? "vertical" : "horizontal");

  if (!sram1_power_on()) return 1;
  uint8_t* const g_input = g_slot[0];

#ifdef TRAFFIC_BENCHMARK
  // Benchmark build (define TRAFFIC_BENCHMARK): no camera, display or SDS; the
  // test image through the detector TRAFFIC_BENCHMARK times, the timings in
  // traffic_benchmark.
  {
    int32_t err = detector_init();
    traffic_benchmark.status = err;
    detections_t d{};
    for (int i = 0; err == 0 && i < TRAFFIC_BENCHMARK; ++i) {
      const uint32_t t0 = cycles();
      err = detector_run(test_image, &d);
      const uint32_t t = us(cycles() - t0);
      // The first run loads the method; after a debugger reset of the core
      // alone it can also meet the NPU still busy with the last job (err 35).
      if (i == 0) {
        err = 0;
        continue;
      }
      traffic_benchmark.runs++;
      traffic_benchmark.total_us_sum += t;
      traffic_benchmark.npu_us_sum += d.npu_us;
      if (traffic_benchmark.total_us_min == 0 || t < traffic_benchmark.total_us_min) traffic_benchmark.total_us_min = t;
      if (traffic_benchmark.npu_us_min == 0 || d.npu_us < traffic_benchmark.npu_us_min) traffic_benchmark.npu_us_min = d.npu_us;
      traffic_benchmark.count = d.count;
    }
    traffic_benchmark.status = err;
    printf("benchmark: %lu runs, total min %lu us avg %lu us, NPU min %lu us avg %lu us, %lu vehicles\n",
           static_cast<unsigned long>(traffic_benchmark.runs), static_cast<unsigned long>(traffic_benchmark.total_us_min),
           static_cast<unsigned long>(traffic_benchmark.total_us_sum / (traffic_benchmark.runs ? traffic_benchmark.runs : 1)),
           static_cast<unsigned long>(traffic_benchmark.npu_us_min),
           static_cast<unsigned long>(traffic_benchmark.npu_us_sum / (traffic_benchmark.runs ? traffic_benchmark.runs : 1)),
           static_cast<unsigned long>(traffic_benchmark.count));
    for (;;) osDelay(1000);
  }
#endif

  int32_t status = detector_init();
  if (status != 0) {
    traffic_status.detector_status = status;
    printf("detector: method load failed (err=%ld)\n", static_cast<long>(status));
    return 1;
  }
  tracker_init(kSize, TRAFFIC_LINE_POS, TRAFFIC_LINE_VERTICAL);
#ifdef APP_HAS_JOYSTICK
  if (joystick_init() != 0) {
    printf("joystick: not available\n");
  } else {
    static const osThreadAttr_t attr = {.name = "joystick", .stack_mem = g_joystick_stack,
                                        .stack_size = sizeof(g_joystick_stack), .priority = osPriorityLow};
    osThreadNew(joystick_thread, nullptr, &attr);
  }
#endif

#if TRAFFIC_SE_DIAG
  // Diagnostic (2026-09-25): what the Secure Enclave made of the boot table,
  // and whether it accepts the DEVICE object (the interconnect firewall).
  {
    static SERVICES_toc_data_t toc;
    uint32_t err = 0;
    uint32_t rc = SERVICES_system_get_toc_data(se_services_s_handle, &toc, &err);
    printf("SE toc: rc %lu err %lu, %lu entries\n", static_cast<unsigned long>(rc), static_cast<unsigned long>(err),
           static_cast<unsigned long>(toc.number_of_toc_entries));
    for (uint32_t i = 0; i < toc.number_of_toc_entries && i < SERVICES_NUMBER_OF_TOC_ENTRIES; ++i) {
      const SERVICES_toc_info_t& e = toc.toc_entry[i];
      printf("  %-8.8s v%lx cpu %lu store %08lx load %08lx boot %08lx size %lu flags %08lx %s\n",
             reinterpret_cast<const char*>(e.image_identifier), static_cast<unsigned long>(e.version),
             static_cast<unsigned long>(e.cpu), static_cast<unsigned long>(e.store_address),
             static_cast<unsigned long>(e.load_address), static_cast<unsigned long>(e.boot_address),
             static_cast<unsigned long>(e.image_size), static_cast<unsigned long>(e.flags),
             reinterpret_cast<const char*>(e.flags_string));
    }
    static uint8_t rev[80];
    err = 0;
    rc = SERVICES_get_se_revision(se_services_s_handle, rev, &err);
    printf("SE revision: rc %lu err %lu: %s\n", static_cast<unsigned long>(rc), static_cast<unsigned long>(err), rev);
    err = 0;
    rc = SERVICES_boot_process_toc_entry(se_services_s_handle, reinterpret_cast<const uint8_t*>("DEVICE"), &err);
    printf("SE process DEVICE: rc %lu err %lu\n", static_cast<unsigned long>(rc), static_cast<unsigned long>(err));
  }
#endif

#ifdef APP_HAS_CAMERA
  // Debugging aid: with TRAFFIC_CAMERA_AUTOSTART 0 the camera stays off until
  // the debugger sets traffic_camera_enable before this point.
  const int32_t camera_status = traffic_camera_enable ? camera_init() : -2;
#if CAMERA_RAW8
  const char* camera_kind = "ARX3A0 RAW8 Bayer, demosaiced on the CPU";
#else
  const char* camera_kind = "640x480 RGB565";
#endif
  printf("camera: %s (%ld)\n", camera_status == 0 ? camera_kind : "not available", static_cast<long>(camera_status));
#else
  const int32_t camera_status = -1;
#endif
  traffic_status.camera_status = camera_status;
  const bool camera_on = camera_status == 0;
#ifdef APP_HAS_CAMERA
  if (camera_on) {
    g_slot_event = osEventFlagsNew(nullptr);
    static const osThreadAttr_t attr = {.name = "camera", .stack_mem = g_camera_stack, .stack_size = sizeof(g_camera_stack),
                                        .priority = osPriorityAboveNormal};
    osThreadNew(camera_thread, nullptr, &attr);
  }
#endif

#ifdef APP_HAS_SDS
  rec_play_init(kInputBytes);
#endif

#ifdef APP_HAS_DISPLAY
  for (uint8_t* fb : g_framebuffer) {
    memset(fb, 0, kFrameBytes);
    SCB_CleanDCache_by_Addr(fb, static_cast<int32_t>(kFrameBytes));
  }
  bool display_on = display_init() == 0;
#ifdef TRAFFIC_LCD_MADCTL
  // The panel's memory access control (MADCTL, 0x36) turns or mirrors the
  // picture in the panel itself, before the video stream starts (the board
  // layer sets the value: bit 7 flips rows, bit 6 columns, on top of the 0x01
  // the pack's panel driver writes). Turning the frame buffer on the CPU
  // instead cost 60 ms a frame and starved the NPU.
  if (display_on) DSI_DCS_Short_Write(0x36, TRAFFIC_LCD_MADCTL);
#endif
  if (display_on) display_on = display_start(g_framebuffer[1]) == 0;
  if (!display_on) printf("display: not available\n");
  int back = 0;
  if (display_on) {
    g_display_go = osSemaphoreNew(1, 0, nullptr);
    g_display_idle = osSemaphoreNew(1, 1, nullptr);
    // Below the vision thread: when the NPU is done, the vision thread goes on at once.
    static const osThreadAttr_t attr = {.name = "display", .stack_mem = g_display_stack,
                                        .stack_size = sizeof(g_display_stack), .priority = osPriorityBelowNormal};
    osThreadNew(display_thread, nullptr, &attr);
  }
#endif

  if (!camera_on) memcpy(g_input, test_image, kInputBytes);

  uint32_t fps_frames = 0, fps_start = osKernelGetTickCount();
  float fps = 0.0f;
  detections_t det{};
  int last_mode = 0;
  for (uint32_t frame = 0;; ++frame) {
    const uint32_t t_frame = cycles();
    int mode = 0;
#ifdef APP_HAS_SDS
    mode = rec_play_poll();
#endif
#ifdef APP_HAS_CAMERA
    if (camera_on && (mode == REC_PLAY_PLAYBACK) != (last_mode == REC_PLAY_PLAYBACK)) camera_pause(mode == REC_PLAY_PLAYBACK);
#endif
    last_mode = mode;
    if (traffic_reset_counts) {
      traffic_reset_counts = 0;
      tracker_reset_counts();
    }
    uint32_t timeslot = osKernelGetTickCount();

    // 1. The input.
    Source source = camera_on ? kCamera : kTestImage;
    const uint8_t* input = g_input;
    uint32_t t0 = cycles();
#ifdef APP_HAS_SDS
    if (mode == REC_PLAY_PLAYBACK) {
      if (rec_play_read_input(g_input, kInputBytes, &timeslot) != 1) {
        // Ended: the streams close at the next poll, once the display thread
        // has put the last frame on the panel (and into the Panel stream).
#ifdef APP_HAS_DISPLAY
        if (display_on) {
          osSemaphoreAcquire(g_display_idle, osWaitForever);
          osSemaphoreRelease(g_display_idle);
        }
#endif
        continue;
      }
      source = kPlayback;
    }
#endif
#ifdef APP_HAS_CAMERA
    if (source == kCamera) {
      input = camera_input(200);  // the camera thread converted it already
      if (input == nullptr) {
        printf("camera: no frame for 200 ms (%lu frames, %lu errors)\n", static_cast<unsigned long>(camera_frame_count()),
               static_cast<unsigned long>(camera_error_count()));
        continue;
      }
    }
#endif
    const uint32_t t_input = cycles() - t0;

    // 2. Recording: the input.
    uint32_t t_sds = 0;
#ifdef APP_HAS_SDS
    if (mode == REC_PLAY_RECORD) {
      t0 = cycles();
      rec_play_write_input(input, kInputBytes, timeslot);
      t_sds += cycles() - t0;
    }
#endif

    // 3. The detector.
    t0 = cycles();
    status = detector_run(input, &det);
    const uint32_t t_detect = cycles() - t0;
    det.frame = frame;
    if (status != 0) {
      traffic_status.detector_status = status;
      printf("detector: run failed (err=%ld)\n", static_cast<long>(status));
      osDelay(1000);
      continue;
    }

    // 4. The tracker and the line. The joystick thread asks for a line
    //    position; the tracker takes it here, between two updates.
    t0 = cycles();
#ifdef APP_HAS_JOYSTICK
    if (g_line_req_pos != tracker_line_pos() || g_line_req_vertical != tracker_line_vertical()) {
      tracker_set_line(g_line_req_pos, g_line_req_vertical);
    }
#endif
    tracker_update(&det);
    const uint32_t t_track = cycles() - t0;

    // 5. The result, while recording or playing back.
#ifdef APP_HAS_SDS
    if (mode != REC_PLAY_IDLE) {
      t0 = cycles();
      rec_play_write_output(&det, sizeof(det), timeslot);
      t_sds += cycles() - t0;
    }
#endif

    // 6. The panel: the picture here, the rest in the display thread while
    //    the NPU works on the next frame.
    t0 = cycles();
#ifdef APP_HAS_DISPLAY
    if (display_on) {
      osSemaphoreAcquire(g_display_idle, osWaitForever);  // the back buffer is off the panel
      uint8_t* fb = g_framebuffer[back];
      image_input_to_view(input, kSize, fb);
      g_job.fb = fb;
      g_job.det = det;
      memcpy(g_job.tracks, tracker_tracks(), sizeof(g_job.tracks));
      g_job.counts = *tracker_counts();
      g_job.live = tracker_live(&g_job.confirmed);
      g_job.has_scores = detector_last_scores(&g_job.scores) == 0 && g_job.scores.anchors <= kAnchors;
      if (g_job.has_scores) {
        memcpy(g_job_scores, g_job.scores.score, static_cast<size_t>(g_job.scores.anchors));
        g_job.scores.score = g_job_scores;
      }
      g_job.source = source;
      g_job.mode = mode;
      g_job.timeslot = timeslot;
      g_job.fps = fps;
      osSemaphoreRelease(g_display_go);
      back ^= 1;
    }
#endif
    const uint32_t t_picture = cycles() - t0;

    // The status for the debugger, the console every 100 frames.
    ++fps_frames;
    const uint32_t now = osKernelGetTickCount();
    if (now - fps_start >= 1000U) {
      fps = fps_frames * 1000.0f / static_cast<float>(now - fps_start);
      fps_frames = 0;
      fps_start = now;
    }
    traffic_status.frames = frame + 1;
    traffic_status.source = source;
    traffic_status.mode = mode;
#ifdef APP_HAS_CAMERA
    traffic_status.camera_frames = camera_frame_count();
    traffic_status.camera_errors = camera_error_count();
    traffic_status.camera_gain = camera_gain();
    traffic_status.camera_mean = g_camera_mean;
    image_wb_gains(const_cast<float*>(traffic_status.wb_gain));
    traffic_status.convert_us = us(g_convert_cycles);
#endif
    traffic_status.detector_status = 0;
    traffic_status.frame_us = us(cycles() - t_frame);
    traffic_status.input_us = us(t_input);
    traffic_status.detect_us = us(t_detect);
    traffic_status.npu_us = det.npu_us;
    traffic_status.track_us = us(t_track);
    traffic_status.picture_us = us(t_picture);
    traffic_status.sds_us = us(t_sds);
    traffic_status.fps = fps;
    traffic_status.tracks = tracker_live(const_cast<uint32_t*>(&traffic_status.tracks_confirmed));
    memcpy(const_cast<traffic_counts_t*>(&traffic_status.counts), tracker_counts(), sizeof(traffic_counts_t));
    memcpy(const_cast<detections_t*>(&traffic_status.result), &det, sizeof(det));
    for (int y = 0; y < 52; ++y)
      for (int x = 0; x < 52; ++x) memcpy(&traffic_thumbnail[(y * 52 + x) * 3], &input[((y * 8) * kSize + x * 8) * 3], 3);
#ifdef APP_HAS_CAMERA
    if (source == kCamera) camera_input_done();
#endif
    if (frame % 100 == 0) {
      const traffic_counts_t* c = tracker_counts();
      printf("frame %lu: %lu vehicle%s, %lu counted (%lu/%lu), %.1f fps; frame %lu us = input %lu + detect %lu (NPU %lu) + track %lu + picture %lu + SDS %lu; display %lu, convert %lu\n",
             static_cast<unsigned long>(frame), static_cast<unsigned long>(det.count), det.count == 1 ? "" : "s",
             static_cast<unsigned long>(c->total), static_cast<unsigned long>(c->by_direction[0]),
             static_cast<unsigned long>(c->by_direction[1]), static_cast<double>(fps),
             static_cast<unsigned long>(traffic_status.frame_us), static_cast<unsigned long>(traffic_status.input_us),
             static_cast<unsigned long>(traffic_status.detect_us), static_cast<unsigned long>(det.npu_us),
             static_cast<unsigned long>(traffic_status.track_us), static_cast<unsigned long>(traffic_status.picture_us),
             static_cast<unsigned long>(traffic_status.sds_us), static_cast<unsigned long>(traffic_status.display_us),
             static_cast<unsigned long>(traffic_status.convert_us));
    }
  }
}
