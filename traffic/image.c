/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pixel work of the traffic counter on the CPU, see image.h.
 *
 * RAW8 Bayer as the CPI stores the ARX3A0's frames: one byte per pixel, the
 * colour of each pixel given by its position in the 2x2 mosaic. RGB565 as the
 * CPI stores an MT9M114's frames: one halfword per pixel, B in bits 4:0, G in
 * 10:5, R in 15:11. The model input is RGB888, three bytes per pixel, R
 * first. The panel frame is the CDC200's RGB888, one 24-bit word per pixel
 * with R in bits 23:16 and B in 7:0 (HWRM, CDC_Ln_PIX_FORMAT), so in memory
 * B first: image_input_to_view() and image_fill() write B, G, R.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "image.h"

static inline void rgb565_to_888(uint16_t p, uint8_t *d)
{
    const uint32_t r = (p >> 11) & 0x1FU, g = (p >> 5) & 0x3FU, b = p & 0x1FU;
    d[0] = (uint8_t)((r << 3) | (r >> 2));
    d[1] = (uint8_t)((g << 2) | (g >> 4));
    d[2] = (uint8_t)((b << 3) | (b >> 2));
}

/* Where output pixel (x, y) of a size x size picture turned `quarter_turns`
   times counter-clockwise lands: the index into the turned picture. */
static inline int turned_index(int x, int y, int size, int quarter_turns)
{
    switch (quarter_turns & 3) {
    case 1:  return (size - 1 - x) * size + y;
    case 2:  return (size - 1 - y) * size + (size - 1 - x);
    case 3:  return x * size + (size - 1 - y);
    default: return y * size + x;
    }
}

/* White balance of the raw frame: no ISP on the E7, so the CPU does a
   gray-world balance on the way to the model input: per colour a lookup
   table with the gain that makes the channel means equal, a tenth of the way
   per frame. */
static uint8_t wb_lut[3][256];
static float   wb_gain[3];

static void wb_build(void)
{
    for (int c = 0; c < 3; c++) {
        for (int v = 0; v < 256; v++) {
            const int o = (int)((float)v * wb_gain[c] + 0.5f);
            wb_lut[c][v] = (uint8_t)(o > 255 ? 255 : o);
        }
    }
}

static void wb_adapt(uint32_t sr, uint32_t sg, uint32_t sb)
{
    /* Next frame's gains: green stays, red and blue move a tenth of the way
       towards the green mean, within 0.5 .. 2. */
    const float mg = (float)sg + 1.0f;
    const float target[3] = {mg / ((float)sr + 1.0f), 1.0f, mg / ((float)sb + 1.0f)};
    for (int c = 0; c < 3; c += 2) {
        float t = target[c] < 0.5f ? 0.5f : (target[c] > 2.0f ? 2.0f : target[c]);
        wb_gain[c] += 0.1f * (t - wb_gain[c]);
    }
    wb_build();
}

void image_wb_gains(float gains[3])
{
    for (int c = 0; c < 3; c++) {
        gains[c] = wb_gain[c];
    }
}

void image_bayer_to_input(const uint8_t *raw, int w, int h, int bayer, uint8_t *rgb, int size, int quarter_turns,
                          uint32_t *mean)
{
    if (wb_gain[1] == 0.0f) {
        wb_gain[0] = wb_gain[1] = wb_gain[2] = 1.0f;
        wb_build();
    }
    /* Offsets of R and B within a 2x2 cell (dx, dy); the two greens are the
       other two pixels. */
    int rx, ry;
    switch (bayer) {
    case IMAGE_BAYER_GRBG: rx = 1; ry = 0; break;
    case IMAGE_BAYER_GBRG: rx = 0; ry = 1; break;
    case IMAGE_BAYER_BGGR: rx = 1; ry = 1; break;
    default:               rx = 0; ry = 0; break;  /* RGGB */
    }
    const int bx = 1 - rx, by = 1 - ry;
    const int side = w < h ? w : h;
    const int x0 = (w - side) / 2, y0 = (h - side) / 2;
    const uint8_t *lr = wb_lut[0], *lg = wb_lut[1], *lb = wb_lut[2];
    uint32_t sr = 0U, sg = 0U, sb = 0U;

    /* Source column of each output column: the even column of its cell. */
    static uint16_t col[1024];
    for (int x = 0; x < size; x++) {
        col[x] = (uint16_t)((x0 + x * side / size) & ~1);
    }
    for (int y = 0; y < size; y++) {
        const int sy = (y0 + y * side / size) & ~1;
        const uint8_t *row0 = raw + sy * w, *row1 = row0 + w;
        const uint8_t *rrow = ry ? row1 : row0, *brow = by ? row1 : row0;
        /* the greens: the other two pixels of the cell */
        const uint8_t *g0row = row0, *g1row = row1;
        const int g0x = (ry == 0) ? 1 - rx : rx;   /* green in row 0: the pixel that is not R (row 0) or not B */
        const int g1x = 1 - g0x;
        uint8_t *d = rgb + turned_index(0, y, size, quarter_turns) * 3;
        const int dstep = (quarter_turns & 3) == 0 ? 3 : (turned_index(1, y, size, quarter_turns) - turned_index(0, y, size, quarter_turns)) * 3;
        for (int x = 0; x < size; x++, d += dstep) {
            const int sx = col[x];
            const uint32_t vr = rrow[sx + rx];
            const uint32_t vb = brow[sx + bx];
            const uint32_t vg = (g0row[sx + g0x] + g1row[sx + g1x] + 1U) >> 1;
            sr += vr;
            sg += vg;
            sb += vb;
            d[0] = lr[vr];
            d[1] = lg[vg];
            d[2] = lb[vb];
        }
    }
    if (mean != NULL) {
        *mean = sg / (uint32_t)(size * size);
    }
    wb_adapt(sr, sg, sb);
}

void image_rgb565_to_input(const uint16_t *camera, int w, int h, uint8_t *rgb, int size, int quarter_turns)
{
    const int side = w < h ? w : h;
    const int x0 = (w - side) / 2, y0 = (h - side) / 2;
    /* Source column of each output column, and one destination step per row
       (as in image_bayer_to_input): no division or index per pixel. */
    static uint16_t col[1024];
    for (int x = 0; x < size; x++) {
        col[x] = (uint16_t)(x0 + x * side / size);
    }
    for (int y = 0; y < size; y++) {
        const uint16_t *row = camera + (y0 + y * side / size) * w;
        uint8_t *d = rgb + turned_index(0, y, size, quarter_turns) * 3;
        const int dstep = (quarter_turns & 3) == 0 ? 3 : (turned_index(1, y, size, quarter_turns) - turned_index(0, y, size, quarter_turns)) * 3;
        for (int x = 0; x < size; x++, d += dstep) {
            rgb565_to_888(row[col[x]], d);
        }
    }
}

/* The AppKit-E7's panel hangs upside down and its controller ignores the
   MADCTL turn bits in video mode: with IMAGE_PANEL_TURN_180 everything drawn
   into the panel buffer goes to the opposite x and y, here and in image_fill,
   which every other drawing routine (boxes, line, score maps, text) uses. */
#ifndef IMAGE_PANEL_TURN_180
#define IMAGE_PANEL_TURN_180 0
#endif

void image_input_to_view(const uint8_t *rgb, int size, uint8_t *panel)
{
    /* 1:1, centred in the view, no scaling. The margins stay as they are. */
    const int x0 = (IMAGE_VIEW - size) / 2, y0 = IMAGE_VIEW_TOP + (IMAGE_VIEW - size) / 2;
    for (int y = 0; y < size; y++) {
        const uint8_t *s = rgb + y * size * 3;
#if IMAGE_PANEL_TURN_180
        /* Row y of the picture lands on panel row H-1-(y0+y), pixels reversed. */
        uint8_t *d = panel + ((IMAGE_PANEL_H - 1 - (y0 + y)) * IMAGE_PANEL_W + (IMAGE_PANEL_W - 1 - x0)) * 3;
        const int step = -3;
#else
        uint8_t *d = panel + ((y0 + y) * IMAGE_PANEL_W + x0) * 3;
        const int step = 3;
#endif
        for (int x = 0; x < size; x++, s += 3, d += step) {
            d[0] = s[2]; /* R, G, B to the panel's B, G, R */
            d[1] = s[1];
            d[2] = s[0];
        }
    }
}

uint32_t image_class_colour(int cls)
{
    static const uint32_t colours[DETECTOR_CLASSES] = {
        0x00E0FFU, /* bicycle: cyan */
        0x00FF40U, /* car: green */
        0xFF40FFU, /* motorcycle: magenta */
        0xFFE000U, /* bus: yellow */
        0xFF8000U, /* truck: orange */
    };
    return cls >= 0 && cls < DETECTOR_CLASSES ? colours[cls] : 0xFFFFFFU;
}

static uint32_t dimmed(uint32_t rgb)
{
    return (rgb >> 1) & 0x7F7F7FU;
}

static void box(uint8_t *panel, int x1, int y1, int x2, int y2, int t, uint32_t rgb)
{
    image_fill(panel, x1, y1, x2, y1 + t, rgb);
    image_fill(panel, x1, y2 - t, x2, y2, rgb);
    image_fill(panel, x1, y1, x1 + t, y2, rgb);
    image_fill(panel, x2 - t, y1, x2, y2, rgb);
}

void image_draw_tracks(uint8_t *panel, const track_t *tracks, int size)
{
    const int x0 = (IMAGE_VIEW - size) / 2, y0 = IMAGE_VIEW_TOP + (IMAGE_VIEW - size) / 2;
    char label[24];
    for (int i = 0; i < TRACKER_MAX_TRACKS; i++) {
        const track_t *t = &tracks[i];
        if (t->id == 0U || t->misses != 0U) {
            continue;
        }
        const uint32_t colour = t->confirmed ? image_class_colour(t->cls) : dimmed(image_class_colour(t->cls));
        int x1 = x0 + (int)t->x1, x2 = x0 + (int)t->x2, y1 = y0 + (int)t->y1, y2 = y0 + (int)t->y2;
        x1 = x1 < x0 ? x0 : x1;
        y1 = y1 < y0 ? y0 : y1;
        x2 = x2 > x0 + size ? x0 + size : x2;
        y2 = y2 > y0 + size ? y0 + size : y2;
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }
        box(panel, x1, y1, x2, y2, t->confirmed ? 3 : 1, colour);
        snprintf(label, sizeof(label), "%lu %s%s", (unsigned long)t->id, DETECTOR_CLASS_NAMES[t->cls], t->counted ? " +" : "");
        /* Above the box, or inside it at the top edge; shifted left and up to
           stay inside the picture: the margins around it are never redrawn,
           so a label that ran into them stayed there. */
        const int lw = 12 * (int)strlen(label) + 2;
        int lx = x1 + lw <= x0 + size ? x1 : x0 + size - lw;
        lx = lx < x0 ? x0 : lx;
        int ly = y1 - 16 >= y0 + 1 ? y1 - 16 : y1 + 4;
        ly = ly + 15 <= y0 + size ? ly : y0 + size - 15;
        image_fill(panel, lx, ly - 1, lx + lw, ly + 15, 0x000000U);
        image_text(panel, lx + 2, ly, 2, label, colour);
    }
}

void image_draw_detections(uint8_t *panel, const detections_t *det, int size)
{
    const int x0 = (IMAGE_VIEW - size) / 2, y0 = IMAGE_VIEW_TOP + (IMAGE_VIEW - size) / 2;
    for (uint32_t i = 0; i < det->count; i++) {
        const detection_t *d = &det->det[i];
        int x1 = x0 + (int)d->x1, x2 = x0 + (int)d->x2, y1 = y0 + (int)d->y1, y2 = y0 + (int)d->y2;
        x1 = x1 < x0 ? x0 : x1;
        y1 = y1 < y0 ? y0 : y1;
        x2 = x2 > x0 + size ? x0 + size : x2;
        y2 = y2 > y0 + size ? y0 + size : y2;
        if (x2 > x1 && y2 > y1) {
            box(panel, x1, y1, x2, y2, 1, dimmed(image_class_colour(d->cls)));
        }
    }
}

void image_draw_line(uint8_t *panel, int size, int pos, int vertical, uint32_t rgb)
{
    const int x0 = (IMAGE_VIEW - size) / 2, y0 = IMAGE_VIEW_TOP + (IMAGE_VIEW - size) / 2;
    /* dashed: 12 on, 6 off */
    for (int i = 0; i < size; i += 18) {
        const int end = i + 12 < size ? i + 12 : size;
        if (vertical) {
            image_fill(panel, x0 + pos - 1, y0 + i, x0 + pos + 2, y0 + end, rgb);
        } else {
            image_fill(panel, x0 + i, y0 + pos - 1, x0 + end, y0 + pos + 2, rgb);
        }
    }
}

void image_fill(uint8_t *panel, int x0, int y0, int x1, int y1, uint32_t rgb)
{
    const uint8_t r = (uint8_t)(rgb >> 16), g = (uint8_t)(rgb >> 8), b = (uint8_t)rgb;
    x0 = x0 < 0 ? 0 : x0;
    y0 = y0 < 0 ? 0 : y0;
    x1 = x1 > IMAGE_PANEL_W ? IMAGE_PANEL_W : x1;
    y1 = y1 > IMAGE_PANEL_H ? IMAGE_PANEL_H : y1;
#if IMAGE_PANEL_TURN_180
    {   /* the same rectangle at the opposite x and y */
        const int tx0 = IMAGE_PANEL_W - x1, ty0 = IMAGE_PANEL_H - y1;
        x1 = IMAGE_PANEL_W - x0;
        y1 = IMAGE_PANEL_H - y0;
        x0 = tx0;
        y0 = ty0;
    }
#endif
    for (int y = y0; y < y1; y++) {
        uint8_t *p = panel + (y * IMAGE_PANEL_W + x0) * 3;
        for (int x = x0; x < x1; x++, p += 3) {
            p[0] = b; /* the panel's byte order */
            p[1] = g;
            p[2] = r;
        }
    }
}

void image_score_maps(uint8_t *panel, int x, int y, const detector_scores_t *scores, int size, int px)
{
    /* Colour per int8 score: black, blue, red, yellow; white at the threshold and above. */
    uint32_t lut[256];
    for (int q = -128; q < 128; q++) {
        float v = ((float)(q - scores->zero_point) * scores->scale) / scores->threshold;
        v = v < 0.0f ? 0.0f : v;
        uint32_t c;
        if (v >= 1.0f) {
            c = 0xFFFFFFU;
        } else {
            const int r = (int)(255.0f * (v < 0.25f ? 0.0f : (v < 0.6f ? (v - 0.25f) / 0.35f : 1.0f)));
            const int gr = (int)(255.0f * (v < 0.6f ? 0.0f : (v - 0.6f) / 0.4f));
            const int bl = (int)(255.0f * (v < 0.25f ? v / 0.25f : (v < 0.6f ? 1.0f - (v - 0.25f) / 0.35f : 0.0f)));
            c = ((uint32_t)r << 16) | ((uint32_t)gr << 8) | (uint32_t)bl;
        }
        lut[(uint8_t)q] = c;
    }
    /* The stride 8, 16 and 32 maps side by side, each drawn `px`-per-cell times
       the stride / 8, so all three are the same size. */
    int base = 0;
    for (int stride = 8; stride <= 32; stride *= 2) {
        const int cells = size / stride, cell = px * stride / 8;
        for (int cy = 0; cy < cells; cy++) {
            for (int cx = 0; cx < cells; cx++) {
                const int x0 = x + cx * cell, y0 = y + cy * cell;
                image_fill(panel, x0, y0, x0 + cell, y0 + cell, lut[(uint8_t)scores->score[base + cy * cells + cx]]);
            }
        }
        base += cells * cells;
        x += cells * cell + px * 4;
    }
}

/* The classic 5x7 font, ASCII 0x20 to 0x5A: five columns per glyph, bit 0 at the top. */
static const uint8_t font5x7[][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x5F, 0x00, 0x00}, {0x00, 0x07, 0x00, 0x07, 0x00},
    {0x14, 0x7F, 0x14, 0x7F, 0x14}, {0x24, 0x2A, 0x7F, 0x2A, 0x12}, {0x23, 0x13, 0x08, 0x64, 0x62},
    {0x36, 0x49, 0x55, 0x22, 0x50}, {0x00, 0x05, 0x03, 0x00, 0x00}, {0x00, 0x1C, 0x22, 0x41, 0x00},
    {0x00, 0x41, 0x22, 0x1C, 0x00}, {0x08, 0x2A, 0x1C, 0x2A, 0x08}, {0x08, 0x08, 0x3E, 0x08, 0x08},
    {0x00, 0x50, 0x30, 0x00, 0x00}, {0x08, 0x08, 0x08, 0x08, 0x08}, {0x00, 0x60, 0x60, 0x00, 0x00},
    {0x20, 0x10, 0x08, 0x04, 0x02}, {0x3E, 0x51, 0x49, 0x45, 0x3E}, {0x00, 0x42, 0x7F, 0x40, 0x00},
    {0x42, 0x61, 0x51, 0x49, 0x46}, {0x21, 0x41, 0x45, 0x4B, 0x31}, {0x18, 0x14, 0x12, 0x7F, 0x10},
    {0x27, 0x45, 0x45, 0x45, 0x39}, {0x3C, 0x4A, 0x49, 0x49, 0x30}, {0x01, 0x71, 0x09, 0x05, 0x03},
    {0x36, 0x49, 0x49, 0x49, 0x36}, {0x06, 0x49, 0x49, 0x29, 0x1E}, {0x00, 0x36, 0x36, 0x00, 0x00},
    {0x00, 0x56, 0x36, 0x00, 0x00}, {0x00, 0x08, 0x14, 0x22, 0x41}, {0x14, 0x14, 0x14, 0x14, 0x14},
    {0x41, 0x22, 0x14, 0x08, 0x00}, {0x02, 0x01, 0x51, 0x09, 0x06}, {0x32, 0x49, 0x79, 0x41, 0x3E},
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, {0x7F, 0x49, 0x49, 0x49, 0x36}, {0x3E, 0x41, 0x41, 0x41, 0x22},
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, {0x7F, 0x49, 0x49, 0x49, 0x41}, {0x7F, 0x09, 0x09, 0x01, 0x01},
    {0x3E, 0x41, 0x41, 0x51, 0x32}, {0x7F, 0x08, 0x08, 0x08, 0x7F}, {0x00, 0x41, 0x7F, 0x41, 0x00},
    {0x20, 0x40, 0x41, 0x3F, 0x01}, {0x7F, 0x08, 0x14, 0x22, 0x41}, {0x7F, 0x40, 0x40, 0x40, 0x40},
    {0x7F, 0x02, 0x04, 0x02, 0x7F}, {0x7F, 0x04, 0x08, 0x10, 0x7F}, {0x3E, 0x41, 0x41, 0x41, 0x3E},
    {0x7F, 0x09, 0x09, 0x09, 0x06}, {0x3E, 0x41, 0x51, 0x21, 0x5E}, {0x7F, 0x09, 0x19, 0x29, 0x46},
    {0x46, 0x49, 0x49, 0x49, 0x31}, {0x01, 0x01, 0x7F, 0x01, 0x01}, {0x3F, 0x40, 0x40, 0x40, 0x3F},
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, {0x7F, 0x20, 0x18, 0x20, 0x7F}, {0x63, 0x14, 0x08, 0x14, 0x63},
    {0x03, 0x04, 0x78, 0x04, 0x03}, {0x61, 0x51, 0x49, 0x45, 0x43},
};

int image_text(uint8_t *panel, int x, int y, int scale, const char *text, uint32_t rgb)
{
    for (; *text != '\0'; text++) {
        int c = (unsigned char)*text;
        if (c >= 'a' && c <= 'z') {
            c -= 'a' - 'A';
        }
        if (c < 0x20 || c > 0x5A) {
            c = '?';
        }
        const uint8_t *glyph = font5x7[c - 0x20];
        for (int col = 0; col < 5; col++) {
            for (int row = 0; row < 7; row++) {
                if (glyph[col] & (1U << row)) {
                    image_fill(panel, x + col * scale, y + row * scale, x + (col + 1) * scale, y + (row + 1) * scale, rgb);
                }
            }
        }
        x += 6 * scale;
    }
    return x;
}
