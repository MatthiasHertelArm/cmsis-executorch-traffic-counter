/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The attention core of YOLO26n's two PSA blocks on the Cortex-M55 (Helium),
 * between the NPU methods stem, mid and head (model/traffic.py,
 * TRAFFIC_ATTENTION=cpu). The Ethos-U55 has no matrix multiply: Vela runs
 * these two products per block as broadcast multiplies with int32 results and
 * row sums, 8 bytes of SRAM traffic per MAC and about 50 of the 80 ms of the
 * whole network; here they are 11 M int8 MACs.
 */
#ifndef TRAFFIC_ATTENTION_H_
#define TRAFFIC_ATTENTION_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ATTENTION_HEADS 2
#define ATTENTION_KEY_DIM 32
#define ATTENTION_HEAD_DIM 64
#define ATTENTION_MAX_TOKENS 256 /* a 16 x 16 stride-32 map: inputs up to 512 x 512 */

/* Affine int8 quantization: real = (q - zero_point) * scale. */
typedef struct {
    float scale;
    int32_t zero_point;
} attention_qparams_t;

/* o = softmax(q k^T / sqrt(key_dim)) v for each head, what
   model/traffic.py attention_core_int8() models on the host.
   qkv: `tokens` rows (NHWC pixels) of ATTENTION_HEADS * (2 key_dim + head_dim)
        int8 channels, per head q | k | v, quantized with `in`.
   out: `tokens` rows of ATTENTION_HEADS * head_dim int8 channels, per head
        its head_dim outputs, quantized with `out_q`.
   Returns 0, or -1 when `tokens` exceeds ATTENTION_MAX_TOKENS. */
int32_t attention_core(const int8_t *qkv, int tokens, attention_qparams_t in, int8_t *out, attention_qparams_t out_q);

#ifdef __cplusplus
}
#endif

#endif /* TRAFFIC_ATTENTION_H_ */
