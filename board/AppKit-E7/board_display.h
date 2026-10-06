/*---------------------------------------------------------------------------
 * Copyright (c) 2026 Arm Limited (or its affiliates). All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AppKit-E7 display: the CDC200 display controller driving the ILI9806E
 * 480x800 panel over MIPI DSI, configured by RTE_Device.h (RGB888, 60 fps).
 * A thin wrapper over the pack's Driver_CDC200 for a double-buffered app.
 *---------------------------------------------------------------------------*/
#ifndef BOARD_DISPLAY_H_
#define BOARD_DISPLAY_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bring up the controller, the DSI link and the panel; 0 on success. */
int32_t display_init(void);

/* Start scanning out from fb (RGB888, APP_DISPLAY_WIDTH x APP_DISPLAY_HEIGHT); 0 on success. */
int32_t display_start(const void *fb);

/* Switch to fb at the next vertical blanking; 0 on success. */
int32_t display_present(const void *fb);

/* Number of frames the controller has started scanning out (SCANLINE0 events). */
uint32_t display_frame_count(void);

/* Block until the frame count exceeds `count` (i.e. one more frame has started),
   with a timeout of a few frames; returns 0 when it did, -1 on timeout. */
int32_t display_wait_frame(uint32_t count);

/* Block until the controller scans out `fb` (the buffer of the last
   display_present()): the buffer shown before it is free from then on.
   0, or -1 on a timeout of a few frames. */
int32_t display_wait_shown(const void *fb);

#ifdef __cplusplus
}
#endif

#endif /* BOARD_DISPLAY_H_ */
