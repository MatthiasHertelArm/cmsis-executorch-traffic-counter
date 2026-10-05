/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SDS recording and playback for the traffic counter (SDS-Framework 3.1, SDSIO
 * over the board's User USB or the J-Link's RTT channel 1). The streams:
 *
 *   CameraIn    the model input of each frame, 416x416 RGB888 (the model input size): recorded from
 *               the camera, or read back in playback instead of the camera
 *   Detections  the detector's result of each frame (detections_t), recorded
 *               in both modes (SDSIO-Server names it Detections.<n>.p.sds in
 *               playback)
 *   Panel       only with SDSIO-Server's flag 0 set (key A) when the streams
 *               open: each frame as the panel showed it, the whole 480x800
 *               frame buffer in records of REC_PLAY_PANEL_ROWS rows
 *
 * SDSIO-Server on the host starts and stops them (keys R, P, S; or
 * --playback): a control thread exchanges the flags with it every 100 ms and
 * runs the state machine of the SDS template application; the vision thread
 * opens and closes the streams between frames (rec_play_poll), so a stream is
 * never closed under a read or a write. The display thread writes Panel; a
 * lock keeps the close from cutting into a frame.
 */
#ifndef YOLO_REC_PLAY_H_
#define YOLO_REC_PLAY_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    REC_PLAY_IDLE = 0,  /* no SDSIO-Server, or not streaming: the camera feeds the detector */
    REC_PLAY_RECORD,    /* the camera feeds the detector, input and result are recorded */
    REC_PLAY_PLAYBACK,  /* CameraIn is read back and feeds the detector, the result is recorded */
} rec_play_mode_t;

/* SDS and the control thread. `input_size` is the byte size of one CameraIn record. */
void rec_play_init(uint32_t input_size);

/* Once per frame, from the vision thread: open or close the streams as the
   host asked, and return the mode of this frame. */
rec_play_mode_t rec_play_poll(void);

/* Playback: the next CameraIn record into `buf`, returned at its time: the
   records keep the spacing of their timeslots (ms), a late one goes at once.
   1 when read, 0 when the stream ended (the streams close at the next poll),
   -1 on an error. */
int32_t rec_play_read_input(void *buf, uint32_t size, uint32_t *timeslot);

/* Recording: append one record; waits while the stream buffer is full.
   0 when written, -1 on an error (the streams close at the next poll). */
int32_t rec_play_write_input(const void *buf, uint32_t size, uint32_t timeslot);
int32_t rec_play_write_output(const void *buf, uint32_t size, uint32_t timeslot);

/* The panel frame (`rows` rows of `row_bytes`) into the Panel stream, as
   records of REC_PLAY_PANEL_ROWS rows with the frame's timeslot; waits while
   the stream buffer is full. Nothing when the Panel stream is not open.
   0 when written or not open, -1 on an error. */
#define REC_PLAY_PANEL_ROWS 40U
int32_t rec_play_write_panel(const void *frame, uint32_t rows, uint32_t row_bytes, uint32_t timeslot);

/* SDS state for the display: sdsState, sdsFlags. */
uint32_t rec_play_state(void);
uint32_t rec_play_flags(void);

#ifdef __cplusplus
}
#endif

#endif /* YOLO_REC_PLAY_H_ */
