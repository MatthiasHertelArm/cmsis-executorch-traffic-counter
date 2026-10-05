/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pixel work of the traffic counter on the CPU: the camera frame to the model
 * input, the picture to the panel, boxes, the counting line and text.
 *
 * The panel frame is APP_DISPLAY_WIDTH x APP_DISPLAY_HEIGHT RGB888 (480 x 800,
 * portrait), in the CDC200's byte order B, G, R; colours given as uint32_t are
 * 0xRRGGBB. The picture goes in a square view of the panel width, 480 x 480,
 * centred vertically; above and below it are the status lines.
 */
#ifndef TRAFFIC_IMAGE_H_
#define TRAFFIC_IMAGE_H_

#include <stdint.h>

#include "detector.h"
#include "tracker.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IMAGE_PANEL_W   APP_DISPLAY_WIDTH
#define IMAGE_PANEL_H   APP_DISPLAY_HEIGHT
#define IMAGE_VIEW      APP_DISPLAY_WIDTH                     /* the square picture on the panel */
#define IMAGE_VIEW_TOP  ((APP_DISPLAY_HEIGHT - APP_DISPLAY_WIDTH) / 2)

/* Bayer orders: the colours of a 2x2 cell, top-left first, then its right neighbour. */
#define IMAGE_BAYER_RGGB 0
#define IMAGE_BAYER_GRBG 1
#define IMAGE_BAYER_GBRG 2
#define IMAGE_BAYER_BGGR 3

/* The centre square of a RAW8 Bayer frame (w x h), demosaiced and scaled to
   the model input (RGB888, size x size): each output pixel takes the colours
   of the 2x2 cell at its place in the frame. quarter_turns turns the picture
   that many times 90 degrees counter-clockwise on the way. A gray-world
   white balance is applied (image_wb_gains); *mean gets the average green
   value of the frame before it, for the exposure control. */
void image_bayer_to_input(const uint8_t *raw, int w, int h, int bayer, uint8_t *rgb, int size, int quarter_turns,
                          uint32_t *mean);

/* The centre square of an RGB565 frame (w x h, h <= w), scaled to the model input. */
void image_rgb565_to_input(const uint16_t *camera, int w, int h, uint8_t *rgb, int size, int quarter_turns);

/* The white-balance gains (R, G, B) image_bayer_to_input applies, adapted every frame. */
void image_wb_gains(float gains[3]);

/* A model input (RGB888, size x size), 1:1 in the middle of the view of the panel. */
void image_input_to_view(const uint8_t *rgb, int size, uint8_t *panel);

/* The colour of a class (0xRRGGBB). */
uint32_t image_class_colour(int cls);

/* The tracks on the view (the picture is `size` pixels square, 1:1, centred):
   a box in the class colour, brighter once confirmed, the id and the class at
   the top; counted tracks get a tick. */
void image_draw_tracks(uint8_t *panel, const track_t *tracks, int size);

/* The detections of the frame as thin boxes (input pixels of a size x size input). */
void image_draw_detections(uint8_t *panel, const detections_t *det, int size);

/* The counting line across the picture: at `pos` input pixels, vertical or horizontal. */
void image_draw_line(uint8_t *panel, int size, int pos, int vertical, uint32_t rgb);

/* The best class score of every anchor of the last run as three heat maps
   (stride 8, 16 and 32) side by side from (x, y), each (size / 8) * px pixels
   square with 4 * px between them: dark (low) through blue and red to
   yellow, white at or above the detection threshold. */
void image_score_maps(uint8_t *panel, int x, int y, const detector_scores_t *scores, int size, int px);

/* Filled rectangle on the panel, clipped: [x0, x1) x [y0, y1). */
void image_fill(uint8_t *panel, int x0, int y0, int x1, int y1, uint32_t rgb);

/* Text in a 5x7 font scaled by `scale` (upper case, digits, " .:%/-+"). Returns the x after it. */
int image_text(uint8_t *panel, int x, int y, int scale, const char *text, uint32_t rgb);

#ifdef __cplusplus
}
#endif

#endif /* TRAFFIC_IMAGE_H_ */
