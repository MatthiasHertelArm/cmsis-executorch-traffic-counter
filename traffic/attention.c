/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The attention core of YOLO26n's PSA blocks with Helium; see attention.h.
 *
 * Per head and query row i (what model/traffic.py attention_core_int8() does
 * on the host):
 *   d_ij = sum_d q_id k_jd - zp sum_d k_jd   exact int32; the terms that are
 *                                            the same for the whole row drop
 *                                            out of the softmax
 *   e_ij = exp((d_ij - max_j d_ij) s^2 / sqrt(key_dim))       in (0, 1]
 *   p_ij = round(255 e_ij)                   uint8, relative to the row's
 *                                            largest weight
 *   o_ic = sum_j p_ij (v_jc - zp) / sum_j p_ij, requantized to `out_q`.
 *
 * The head's q, k and v are first copied out of the NHWC tensor into the
 * DTCM: q and k row by row, v transposed and offset to uint8 (v ^ 0x80) so
 * that the weighted sum is one 16-lane unsigned dot product per 16 tokens.
 * exp() is a cubic in 2^frac, good to 0.02 of p's last step.
 * Query rows go in groups of four: each k row and each v column is loaded
 * once per group, and the four dot-product chains are independent.
 */

#include "attention.h"

#include <arm_mve.h>
#include <string.h>

#define KD ATTENTION_KEY_DIM
#define HD ATTENTION_HEAD_DIM
#define GROUP (2 * KD + HD)                   /* q | k | v channels of one head */
#define QKV_CHANNELS (ATTENTION_HEADS * GROUP)
#define OUT_CHANNELS (ATTENTION_HEADS * HD)
#define PADDED ((ATTENTION_MAX_TOKENS + 15) & ~15)
#define ROWS 4                                /* query rows per group */

/* 1 / sqrt(key_dim): Ultralytics Attention.scale. */
static const float kScale = 0.17677669529663687f;

static int8_t s_q[ATTENTION_MAX_TOKENS][KD] __attribute__((aligned(16)));
static int8_t s_k[ATTENTION_MAX_TOKENS][KD] __attribute__((aligned(16)));
static int32_t s_ksum[ATTENTION_MAX_TOKENS] __attribute__((aligned(16)));    /* zp * sum_d k_jd */
static uint8_t s_vt[HD][PADDED] __attribute__((aligned(16)));               /* v ^ 0x80, transposed */
static int32_t s_logit[ROWS][PADDED] __attribute__((aligned(16)));
static uint8_t s_p[ROWS][PADDED] __attribute__((aligned(16)));
static uint32_t s_acc[ROWS][HD] __attribute__((aligned(16)));

/* 255 e^x for x <= 0, with x given as t = x log2(e): 255 2^t = 2^floor(t) *
   255 2^frac(t), the fraction by a cubic (least-squares on [0, 1), relative
   error < 8e-5, 0.02 of the last step of p); below 2^-24 p is 0 anyway. */
static inline float32x4_t exp2_255(float32x4_t t)
{
    t = vmaxnmq_f32(t, vdupq_n_f32(-24.0f));
    const float32x4_t n = vrndmq_f32(t);
    const float32x4_t f = vsubq_f32(t, n);
    float32x4_t p = vdupq_n_f32(0.07790716f * 255.0f);
    p = vfmaq_f32(vdupq_n_f32(0.22623319f * 255.0f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.6957771f * 255.0f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.99992783f * 255.0f), p, f);
    const int32x4_t e = vshlq_n_s32(vaddq_n_s32(vcvtq_s32_f32(n), 127), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

/* One head: copy q, k (and their zero-point term) and v transposed into the DTCM. */
static void load_head(const int8_t *qkv, int tokens, int padded, int h, int32_t zp)
{
    const uint16x8_t column = vmulq_n_u16(vidupq_n_u16(0, 1), PADDED); /* ch * PADDED, ch = 0..7 */
    for (int j = 0; j < tokens; ++j) {
        const int8_t *row = qkv + j * QKV_CHANNELS + h * GROUP;
        const int8x16_t q0 = vldrbq_s8(row), q1 = vldrbq_s8(row + 16);
        const int8x16_t k0 = vldrbq_s8(row + KD), k1 = vldrbq_s8(row + KD + 16);
        vstrbq_s8(s_q[j], q0);
        vstrbq_s8(s_q[j] + 16, q1);
        vstrbq_s8(s_k[j], k0);
        vstrbq_s8(s_k[j] + 16, k1);
        s_ksum[j] = zp * (vaddvq_s8(k0) + vaddvq_s8(k1));
        /* v ^ 0x80 into column j of s_vt, eight channels per scatter. */
        const uint8_t *v = (const uint8_t *)row + 2 * KD;
        for (int ch = 0; ch < HD; ch += 8) {
            vstrbq_scatter_offset_u16(&s_vt[ch][j], column, veorq_u16(vldrbq_u16(v + ch), vdupq_n_u16(0x80)));
        }
    }
    for (int ch = 0; ch < HD; ++ch) {
        memset(&s_vt[ch][tokens], 0, (size_t)(padded - tokens));
    }
}

/* q k^T of two query rows against all keys, into s_logit[r], s_logit[r + 1]; returns the row maxima. */
static void logits2(int i, int tokens, int r, int32_t max[2])
{
    const int8x16_t a0 = vldrbq_s8(s_q[i]), a1 = vldrbq_s8(s_q[i] + 16);
    const int8x16_t b0 = vldrbq_s8(s_q[i + 1]), b1 = vldrbq_s8(s_q[i + 1] + 16);
    int32_t ma = INT32_MIN, mb = INT32_MIN;
    for (int j = 0; j < tokens; ++j) {
        const int8x16_t k0 = vldrbq_s8(s_k[j]), k1 = vldrbq_s8(s_k[j] + 16);
        const int32_t da = vmladavaq_s8(vmladavq_s8(a0, k0), a1, k1) - s_ksum[j];
        const int32_t db = vmladavaq_s8(vmladavq_s8(b0, k0), b1, k1) - s_ksum[j];
        s_logit[r][j] = da;
        s_logit[r + 1][j] = db;
        ma = da > ma ? da : ma;
        mb = db > mb ? db : mb;
    }
    max[0] = ma;
    max[1] = mb;
}

/* p = round(255 exp((d - max) c)) of one row, into s_p[r]; returns sum p. */
static uint32_t weights(int r, int tokens, int padded, int32_t max, float c)
{
    for (int j = tokens; j < padded; ++j) {
        s_logit[r][j] = max - (1 << 24); /* weight 0 */
    }
    const float c2 = c * 1.44269504f; /* to powers of 2 */
    uint32_t sum = 0;
    for (int j = 0; j < padded; j += 4) {
        const float32x4_t t = vmulq_n_f32(vcvtq_f32_s32(vsubq_n_s32(vldrwq_s32(&s_logit[r][j]), max)), c2);
        const uint32x4_t p = vcvtaq_u32_f32(exp2_255(t));
        sum = vaddvaq_u32(sum, p);
        vstrbq_u32(&s_p[r][j], p);
    }
    return sum;
}

int32_t attention_core(const int8_t *qkv, int tokens, attention_qparams_t in, int8_t *out, attention_qparams_t out_q)
{
    if (tokens <= 0 || tokens > ATTENTION_MAX_TOKENS) {
        return -1;
    }
    const int padded = (tokens + 15) & ~15;
    const int32_t zp = in.zero_point;
    const float c = in.scale * in.scale * kScale;
    const int chunks = padded / 16;

    for (int h = 0; h < ATTENTION_HEADS; ++h) {
        load_head(qkv, tokens, padded, h, zp);
        /* Rows past the last token repeat it (computed, not stored). */
        for (int j = tokens; j < ((tokens + ROWS - 1) & ~(ROWS - 1)); ++j) {
            memcpy(s_q[j], s_q[tokens - 1], KD);
        }

        for (int i = 0; i < tokens; i += ROWS) {
            int32_t max[ROWS];
            uint32_t sum[ROWS];
            logits2(i, tokens, 0, &max[0]);
            logits2(i + 2, tokens, 2, &max[2]);
            for (int r = 0; r < ROWS; ++r) {
                sum[r] = weights(r, tokens, padded, max[r], c);
            }

            /* sum_j p_rj (v_jc ^ 0x80) for the four rows, one v column at a time. */
            for (int ch = 0; ch < HD; ++ch) {
                const uint8_t *vt = s_vt[ch];
                uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                for (int k = 0; k < chunks; ++k) {
                    const uint8x16_t v = vldrbq_u8(vt + 16 * k);
                    a0 = vmladavaq_u8(a0, vldrbq_u8(&s_p[0][16 * k]), v);
                    a1 = vmladavaq_u8(a1, vldrbq_u8(&s_p[1][16 * k]), v);
                    a2 = vmladavaq_u8(a2, vldrbq_u8(&s_p[2][16 * k]), v);
                    a3 = vmladavaq_u8(a3, vldrbq_u8(&s_p[3][16 * k]), v);
                }
                s_acc[0][ch] = a0;
                s_acc[1][ch] = a1;
                s_acc[2][ch] = a2;
                s_acc[3][ch] = a3;
            }

            /* o_c = (acc_c - (128 + zp) sum p) s_in / (sum p s_out) + zp_out */
            for (int r = 0; r < ROWS && i + r < tokens; ++r) {
                const int32_t offset = (128 + zp) * (int32_t)sum[r];
                const float f = in.scale / (out_q.scale * (float)sum[r]);
                int8_t *o = out + (i + r) * OUT_CHANNELS + h * HD;
                for (int ch = 0; ch < HD; ch += 4) {
                    int32x4_t a = vsubq_n_s32(vreinterpretq_s32_u32(vldrwq_u32(&s_acc[r][ch])), offset);
                    int32x4_t v = vaddq_n_s32(vcvtaq_s32_f32(vmulq_n_f32(vcvtq_f32_s32(a), f)), out_q.zero_point);
                    v = vmaxq_s32(vminq_s32(v, vdupq_n_s32(127)), vdupq_n_s32(-128));
                    vstrbq_s32(o + ch, v);
                }
            }
        }
    }
    return 0;
}
