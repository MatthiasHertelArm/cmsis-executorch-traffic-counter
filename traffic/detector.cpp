/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The vehicle detector: model/traffic.py, exported by create_ai_layer.py
 * into ai_layer/. It returns, per anchor, the box distances and one
 * score per vehicle class (int8). The CPU does what is left of the YOLO26
 * head: the best class of each anchor, the threshold, the distances of the
 * survivors turned into boxes. The one-to-one head of YOLO26 is NMS-free; an
 * IoU check still drops the rare duplicate.
 *
 * Two exports, told apart by model_io.h:
 *   TRAFFIC_ATTENTION=npu  one method, `detect`, the whole graph on the NPU;
 *   TRAFFIC_ATTENTION=cpu  (the default) three NPU methods, stem, mid and
 *                          head, with the attention cores of the two PSA
 *                          blocks on the CPU in between (attention.c): the
 *                          Ethos-U55 spends about 50 of its 80 ms on them.
 *                          The same tensor has other quantization parameters
 *                          as an output of one method and an input of the
 *                          next; a 256-entry table per tensor requantizes it
 *                          in place.
 *
 * The Ethos-U backend copies the input into the NPU scratch with
 * arm_ethos_io_memcpy(); the override below turns RGB888 into the int8 input
 * on the way (the input scale is 1/255 with zero point -128, so
 * q = pixel - 128 = pixel ^ 0x80): no staging buffer, no conversion pass.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include <arm_mve.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_embedded_module.hpp"
#include "attention.h"
#include "detector.h"
#include "model_io.h"
#include "model_pte.h"

#if defined(MODEL_STEM_METHOD)
#define DETECTOR_SPLIT 1
#define DETECTOR_INPUT_SHAPE MODEL_STEM_INPUT0_SHAPE
#define DETECTOR_INPUT_NUMEL MODEL_STEM_INPUT0_NUMEL
#define DETECTOR_BOX_SHAPE MODEL_HEAD_OUTPUT0_SHAPE
#define DETECTOR_BOX_SCALE MODEL_HEAD_OUTPUT0_SCALE
#define DETECTOR_BOX_ZERO_POINT MODEL_HEAD_OUTPUT0_ZERO_POINT
#define DETECTOR_CLS_SHAPE MODEL_HEAD_OUTPUT1_SHAPE
#define DETECTOR_CLS_SCALE MODEL_HEAD_OUTPUT1_SCALE
#define DETECTOR_CLS_ZERO_POINT MODEL_HEAD_OUTPUT1_ZERO_POINT
#else
#define DETECTOR_SPLIT 0
#define DETECTOR_INPUT_SHAPE MODEL_DETECT_INPUT0_SHAPE
#define DETECTOR_INPUT_NUMEL MODEL_DETECT_INPUT0_NUMEL
#define DETECTOR_BOX_SHAPE MODEL_DETECT_OUTPUT0_SHAPE
#define DETECTOR_BOX_SCALE MODEL_DETECT_OUTPUT0_SCALE
#define DETECTOR_BOX_ZERO_POINT MODEL_DETECT_OUTPUT0_ZERO_POINT
#define DETECTOR_CLS_SHAPE MODEL_DETECT_OUTPUT1_SHAPE
#define DETECTOR_CLS_SCALE MODEL_DETECT_OUTPUT1_SCALE
#define DETECTOR_CLS_ZERO_POINT MODEL_DETECT_OUTPUT1_ZERO_POINT
#endif

using arm::embedded::EmbeddedModule;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Span;

namespace {

// Score threshold of a vehicle (traffic/eval_vehicles.py measures the
// vehicle AP50 of the float and the int8 model at a size).
constexpr float kScoreThreshold = 0.30f;
constexpr float kIouDuplicate = 0.70f;

constexpr int32_t kInputShape[] = DETECTOR_INPUT_SHAPE;  // {1, S, S, 3}
constexpr int kSize = kInputShape[1];
constexpr int32_t kBoxShape[] = DETECTOR_BOX_SHAPE;  // {1, N, 4}
constexpr int kAnchors = kBoxShape[1];
constexpr int32_t kClsShape[] = DETECTOR_CLS_SHAPE;  // {1, N, C}
constexpr int kClasses = kClsShape[2];
constexpr int kStrides[] = {8, 16, 32};
static_assert(kAnchors == (kSize / 8) * (kSize / 8) + (kSize / 16) * (kSize / 16) + (kSize / 32) * (kSize / 32),
              "anchor count");
static_assert(kClasses == DETECTOR_CLASSES, "class count of model/traffic.py");

// Method pool: the runtime's structures and, for `detect`, its planned buffer
// (the two outputs, 32 kB). Temp pool: Vela's scratch, 1220 KiB for YOLO26n
// at 416 x 416 (the largest of stem, mid and head).
#if DETECTOR_SPLIT
constexpr size_t kMethodPoolSize = 0x4000;  // 2 kB used for the three methods
#else
constexpr size_t kMethodPoolSize = 0x40000;
#endif
constexpr size_t kTempPoolSize = 0x140000;
alignas(16) uint8_t g_method_pool[kMethodPoolSize] __attribute__((section(APP_POOL_SECTION)));
#ifndef APP_TEMP_POOL_SECTION
#define APP_TEMP_POOL_SECTION APP_POOL_SECTION
#endif
// The NPU's scratch (Vela's temporary tensors): on the E7 in SRAM1, which is
// as fast as SRAM0 (both 64-bit at 400 MHz) and leaves SRAM0 to the panel
// buffers and the camera frames.
alignas(16) uint8_t g_temp_pool[kTempPoolSize] __attribute__((section(APP_TEMP_POOL_SECTION)));

const uint8_t* g_input_rgb;  // the frame the input copy converts
int8_t g_best[kAnchors];     // the best class score of each anchor of the last run
bool g_have_scores;
uint32_t g_candidates;       // its anchors above the threshold
int8_t g_score_max;          // its highest score
uint32_t g_npu_begin, g_npu_cycles;
uint32_t g_attention_cycles, g_requant_cycles;

inline uint32_t cycles() { return DWT->CYCCNT; }
inline uint32_t us(uint32_t c) { return static_cast<uint32_t>(static_cast<uint64_t>(c) * 1000000U / SystemCoreClock); }

// The module and the input tensor live as long as the application.
alignas(EmbeddedModule) uint8_t g_module_storage[sizeof(EmbeddedModule)];
EmbeddedModule* g_module;
std::array<SizesType, 4> g_sizes;
std::array<DimOrderType, 4> g_dim_order;
alignas(TensorImpl) uint8_t g_input_storage[sizeof(TensorImpl)];
TensorImpl* g_input;

float iou(const detection_t& a, const detection_t& b) {
  const float w = fminf(a.x2, b.x2) - fmaxf(a.x1, b.x1);
  const float h = fminf(a.y2, b.y2) - fmaxf(a.y1, b.y1);
  if (w <= 0.0f || h <= 0.0f) return 0.0f;
  const float inter = w * h;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

// The rest of the YOLO26 head: the best class of each anchor, the anchors
// above the threshold, their distances to boxes. Both outputs are
// anchor-major: box[a * 4 + side], score[a * kClasses + class].
uint32_t decode(const int8_t* box, const int8_t* score, detection_t* out) {
  constexpr float kBoxScale = DETECTOR_BOX_SCALE;
  constexpr int kBoxZp = DETECTOR_BOX_ZERO_POINT;
  constexpr float kScoreScale = DETECTOR_CLS_SCALE;
  constexpr int kScoreZp = DETECTOR_CLS_ZERO_POINT;
  const int threshold_q = static_cast<int>(ceilf(kScoreThreshold / kScoreScale)) + kScoreZp;

  detection_t found[DETECTOR_MAX_DETECTIONS * 2];
  int n = 0;
  int base = 0;
  g_candidates = 0;
  g_score_max = INT8_MIN;
  for (int stride : kStrides) {
    const int cells = kSize / stride;
    for (int a = base; a < base + cells * cells; ++a) {
      const int8_t* scores = score + a * kClasses;
      int8_t best = scores[0];
      int cls = 0;
      for (int c = 1; c < kClasses; ++c) {
        if (scores[c] > best) {
          best = scores[c];
          cls = c;
        }
      }
      g_best[a] = best;
      if (best > g_score_max) g_score_max = best;
      if (best < threshold_q) continue;
      ++g_candidates;
      const int cell = a - base;
      const float cx = static_cast<float>(cell % cells) + 0.5f;
      const float cy = static_cast<float>(cell / cells) + 0.5f;
      const float s = static_cast<float>(stride);
      detection_t d;
      const int8_t* ltrb = box + a * 4;
      d.x1 = (cx - (ltrb[0] - kBoxZp) * kBoxScale) * s;
      d.y1 = (cy - (ltrb[1] - kBoxZp) * kBoxScale) * s;
      d.x2 = (cx + (ltrb[2] - kBoxZp) * kBoxScale) * s;
      d.y2 = (cy + (ltrb[3] - kBoxZp) * kBoxScale) * s;
      d.score = (best - kScoreZp) * kScoreScale;
      d.cls = cls;
      if (n < DETECTOR_MAX_DETECTIONS * 2) {
        found[n++] = d;
      } else {  // full: replace the weakest
        int weakest = 0;
        for (int i = 1; i < n; ++i)
          if (found[i].score < found[weakest].score) weakest = i;
        if (d.score > found[weakest].score) found[weakest] = d;
      }
    }
    base += cells * cells;
  }

  // Strongest first, drop what overlaps a stronger box.
  for (int i = 1; i < n; ++i)
    for (int j = i; j > 0 && found[j].score > found[j - 1].score; --j) std::swap(found[j], found[j - 1]);
  uint32_t kept = 0;
  for (int i = 0; i < n && kept < DETECTOR_MAX_DETECTIONS; ++i) {
    bool duplicate = false;
    for (uint32_t k = 0; k < kept && !duplicate; ++k) duplicate = iou(found[i], out[k]) > kIouDuplicate;
    if (!duplicate) out[kept++] = found[i];
  }
  return kept;
}

// The method allocator, with its high-water mark for the debugger
// (detector_method_pool_used).
class TrackingAllocator : public MemoryAllocator {
 public:
  using MemoryAllocator::MemoryAllocator;
  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    void* p = MemoryAllocator::allocate(size, alignment);
    if (p != nullptr) used = static_cast<size_t>(static_cast<uint8_t*>(p) + size - base_address());
    return p;
  }
  size_t used = 0;
};
TrackingAllocator* g_method_allocator;

#if DETECTOR_SPLIT
constexpr size_t align16(size_t n) { return (n + 15) & ~static_cast<size_t>(15); }

// Planned memory: each method's outputs, 16-byte aligned one after the other
// (the program's memory plan; detector_init checks the sizes). stem's (520 kB
// at 416 x 416) sit in SRAM0 with the pools, and head's reuse them: stem's
// outputs are dead once mid has run. mid's (140 kB) and the two attention
// outputs sit in the DTCM, next to the CPU that requantizes them.
constexpr size_t kStemPlanned = align16(MODEL_STEM_OUTPUT0_NUMEL) + align16(MODEL_STEM_OUTPUT1_NUMEL) +
                                align16(MODEL_STEM_OUTPUT2_NUMEL) + align16(MODEL_STEM_OUTPUT3_NUMEL);
constexpr size_t kMidPlanned = align16(MODEL_MID_OUTPUT0_NUMEL) + align16(MODEL_MID_OUTPUT1_NUMEL) +
                               align16(MODEL_MID_OUTPUT2_NUMEL) + align16(MODEL_MID_OUTPUT3_NUMEL) +
                               align16(MODEL_MID_OUTPUT4_NUMEL);
constexpr size_t kHeadPlanned = align16(MODEL_HEAD_OUTPUT0_NUMEL) + align16(MODEL_HEAD_OUTPUT1_NUMEL);
static_assert(kHeadPlanned <= kStemPlanned, "head's outputs reuse stem's planned memory");
alignas(16) uint8_t g_stem_planned[kStemPlanned] __attribute__((section(APP_POOL_SECTION)));
alignas(16) uint8_t g_mid_planned[kMidPlanned];
Span<uint8_t> g_planned_span[3];
alignas(HierarchicalAllocator) uint8_t g_planned_storage[3][sizeof(HierarchicalAllocator)];

// The attention cores' outputs: input 0 of mid and of head.
alignas(16) int8_t g_o10[MODEL_MID_INPUT0_NUMEL];
alignas(16) int8_t g_o22[MODEL_HEAD_INPUT0_NUMEL];

constexpr int32_t kQkv10Shape[] = MODEL_STEM_OUTPUT3_SHAPE;  // {1, H, W, 256}
constexpr int kTokens10 = kQkv10Shape[1] * kQkv10Shape[2];
constexpr int32_t kQkv22Shape[] = MODEL_MID_OUTPUT0_SHAPE;
constexpr int kTokens22 = kQkv22Shape[1] * kQkv22Shape[2];
static_assert(kQkv10Shape[3] == ATTENTION_HEADS * (2 * ATTENTION_KEY_DIM + ATTENTION_HEAD_DIM), "qkv channels");
static_assert(kQkv22Shape[3] == kQkv10Shape[3], "qkv channels");
static_assert(MODEL_MID_INPUT0_NUMEL == kTokens10 * ATTENTION_HEADS * ATTENTION_HEAD_DIM, "attention output");
static_assert(MODEL_HEAD_INPUT0_NUMEL == kTokens22 * ATTENTION_HEADS * ATTENTION_HEAD_DIM, "attention output");

// An int8 tensor from one method's quantization to the next one's: a table of
// all 256 values, applied in place, 16 lanes per gather.
struct Requant {
  int8_t lut[256];
  bool identity;
  void init(float s_in, int zp_in, float s_out, int zp_out) {
    identity = true;
    for (int q = -128; q < 128; ++q) {
      int v = static_cast<int>(lroundf(static_cast<float>(q - zp_in) * s_in / s_out)) + zp_out;
      v = v < -128 ? -128 : (v > 127 ? 127 : v);
      lut[q + 128] = static_cast<int8_t>(v);
      identity = identity && v == q;
    }
  }
  void apply(int8_t* data, size_t size) const {
    if (identity) return;
    const uint8_t* table = reinterpret_cast<const uint8_t*>(lut);
    uint8_t* d = reinterpret_cast<uint8_t*>(data);
    const uint8x16_t bias = vdupq_n_u8(0x80);
    for (int32_t left = static_cast<int32_t>(size); left > 0; left -= 16, d += 16) {
      const mve_pred16_t pred = vctp8q(static_cast<uint32_t>(left));
      const uint8x16_t index = veorq_u8(vldrbq_z_u8(d, pred), bias);
      vstrbq_p_u8(d, vldrbq_gather_offset_z_u8(table, index, pred), pred);
    }
  }
};
#define REQUANT(dst, from, to) \
  dst.init(from##_SCALE, from##_ZERO_POINT, to##_SCALE, to##_ZERO_POINT); \
  static_assert(from##_NUMEL == to##_NUMEL, #from " -> " #to)
Requant g_l4, g_l6, g_y10, g_qkv10, g_qkv22, g_z22, g_y22, g_box34, g_cls34;

void init_requant() {
  REQUANT(g_l4, MODEL_STEM_OUTPUT0, MODEL_MID_INPUT3);
  REQUANT(g_l6, MODEL_STEM_OUTPUT1, MODEL_MID_INPUT4);
  REQUANT(g_y10, MODEL_STEM_OUTPUT2, MODEL_MID_INPUT2);
  REQUANT(g_qkv10, MODEL_STEM_OUTPUT3, MODEL_MID_INPUT1);
  REQUANT(g_qkv22, MODEL_MID_OUTPUT0, MODEL_HEAD_INPUT1);
  REQUANT(g_z22, MODEL_MID_OUTPUT1, MODEL_HEAD_INPUT2);
  REQUANT(g_y22, MODEL_MID_OUTPUT2, MODEL_HEAD_INPUT3);
  REQUANT(g_box34, MODEL_MID_OUTPUT3, MODEL_HEAD_INPUT4);
  REQUANT(g_cls34, MODEL_MID_OUTPUT4, MODEL_HEAD_INPUT5);
}

// An input tensor of mid or head: shape from model_io.h, data set per run.
struct Input {
  std::array<SizesType, 4> sizes;
  std::array<DimOrderType, 4> dim_order;
  alignas(TensorImpl) uint8_t storage[sizeof(TensorImpl)];
  TensorImpl* impl;
  void init(std::initializer_list<int32_t> shape) {
    int n = 0;
    for (int32_t d : shape) {
      sizes[n] = d;
      dim_order[n] = static_cast<DimOrderType>(n);
      ++n;
    }
    impl = new (storage) TensorImpl(ScalarType::Char, n, sizes.data(), nullptr, dim_order.data());
  }
  EValue operator()(void* data) {
    impl->set_data(data);
    return EValue(Tensor(impl));
  }
};
Input g_mid_in[MODEL_MID_NUM_INPUTS], g_head_in[MODEL_HEAD_NUM_INPUTS];

// Load a method with its planned memory in `buffer`.
int32_t load_method(const char* name, uint8_t* buffer, size_t size, int index) {
  auto meta = g_module->method_meta(name);
  if (!meta.ok()) return static_cast<int32_t>(meta.error());
  if (meta->num_memory_planned_buffers() != 1) return -2;
  auto need = meta->memory_planned_buffer_size(0);
  if (!need.ok() || static_cast<size_t>(*need) > size) return -3;
  g_planned_span[index] = Span<uint8_t>(buffer, size);
  auto* planned = new (g_planned_storage[index]) HierarchicalAllocator(Span<Span<uint8_t>>(&g_planned_span[index], 1));
  return static_cast<int32_t>(g_module->load_method(name, planned));
}

inline int8_t* data(const EValue& v) { return v.toTensor().mutable_data_ptr<int8_t>(); }
#endif  // DETECTOR_SPLIT

}  // namespace

extern "C" {
// For the debugger: the last run's CPU attention and requantization times,
// and the method pool's high-water mark.
volatile uint32_t detector_attention_us, detector_requant_us, detector_method_pool_used;
}

extern "C" const int DETECTOR_INPUT_SIZE = kSize;
extern "C" const int DETECTOR_ANCHORS = kAnchors;
extern "C" const char* const DETECTOR_CLASS_NAMES[DETECTOR_CLASSES] = {"BICYCLE", "CAR", "MOTORCYCLE", "BUS", "TRUCK"};

// The Ethos-U backend's IO copy hook (strong override of the weak default).
// The input: RGB888 to int8 (x ^ 0x80), 16 bytes per Helium load / store,
// tail by predicate. Everything else (the two small outputs) is a memcpy.
extern "C" void arm_ethos_io_memcpy(void* dst, const void* src, size_t size) {
  if (src == g_input_rgb && size == DETECTOR_INPUT_NUMEL) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8x16_t offset = vdupq_n_u8(0x80);
    for (int32_t left = static_cast<int32_t>(size); left > 0; left -= 16, s += 16, d += 16) {
      mve_pred16_t pred = vctp8q(static_cast<uint32_t>(left));
      vstrbq_p_u8(d, veorq_u8(vldrbq_z_u8(s, pred), offset), pred);
    }
    return;
  }
  memcpy(dst, src, size);
}

// The driver's hooks around the NPU job: command stream start to interrupt.
extern "C" void ethosu_inference_begin(struct ethosu_driver* drv, void* user_arg) {
  (void)drv;
  (void)user_arg;
  g_npu_begin = cycles();
}
extern "C" void ethosu_inference_end(struct ethosu_driver* drv, void* user_arg) {
  (void)drv;
  (void)user_arg;
  g_npu_cycles += cycles() - g_npu_begin;
}

extern "C" int32_t detector_init(void) { return detector_init_from(model_pte); }

extern "C" int32_t detector_init_from(const uint8_t* program) {
  executorch::runtime::runtime_init();
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  auto method_allocator = std::make_unique<TrackingAllocator>(kMethodPoolSize, g_method_pool);
  g_method_allocator = method_allocator.get();
  g_module = new (g_module_storage)
      EmbeddedModule(program, model_pte_size, std::make_unique<BufferDataLoader>(program, model_pte_size),
                     std::move(method_allocator), std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));
  for (int i = 0; i < 4; ++i) {
    g_sizes[i] = kInputShape[i];
    g_dim_order[i] = static_cast<DimOrderType>(i);
  }
  g_input = new (g_input_storage) TensorImpl(ScalarType::Char, 4, g_sizes.data(), nullptr, g_dim_order.data());
#if DETECTOR_SPLIT
  init_requant();
  const std::initializer_list<int32_t> mid_shapes[] = {MODEL_MID_INPUT0_SHAPE, MODEL_MID_INPUT1_SHAPE,
                                                       MODEL_MID_INPUT2_SHAPE, MODEL_MID_INPUT3_SHAPE,
                                                       MODEL_MID_INPUT4_SHAPE};
  const std::initializer_list<int32_t> head_shapes[] = {MODEL_HEAD_INPUT0_SHAPE, MODEL_HEAD_INPUT1_SHAPE,
                                                        MODEL_HEAD_INPUT2_SHAPE, MODEL_HEAD_INPUT3_SHAPE,
                                                        MODEL_HEAD_INPUT4_SHAPE, MODEL_HEAD_INPUT5_SHAPE};
  for (int i = 0; i < MODEL_MID_NUM_INPUTS; ++i) g_mid_in[i].init(mid_shapes[i]);
  for (int i = 0; i < MODEL_HEAD_NUM_INPUTS; ++i) g_head_in[i].init(head_shapes[i]);
  int32_t err = load_method(MODEL_STEM_METHOD, g_stem_planned, sizeof(g_stem_planned), 0);
  if (err == 0) err = load_method(MODEL_MID_METHOD, g_mid_planned, sizeof(g_mid_planned), 1);
  if (err == 0) err = load_method(MODEL_HEAD_METHOD, g_stem_planned, kHeadPlanned, 2);
#else
  const int32_t err = static_cast<int32_t>(g_module->load_method(MODEL_DETECT_METHOD));
#endif
  detector_method_pool_used = g_method_allocator->used;
  return err;
}

extern "C" int32_t detector_run(const uint8_t* rgb, detections_t* out) {
  g_input_rgb = rgb;
  g_input->set_data(const_cast<uint8_t*>(rgb));
  std::vector<EValue> inputs{EValue(Tensor(g_input))};

  g_npu_cycles = 0;
  const uint32_t t0 = cycles();
#if DETECTOR_SPLIT
  g_attention_cycles = g_requant_cycles = 0;
  auto stem = g_module->execute(MODEL_STEM_METHOD, inputs);
  if (!stem.ok()) return static_cast<int32_t>(stem.error());
  if (stem->size() != MODEL_STEM_NUM_OUTPUTS) return -1;
  int8_t* const l4 = data((*stem)[0]);
  int8_t* const l6 = data((*stem)[1]);
  int8_t* const y10 = data((*stem)[2]);
  int8_t* const qkv10 = data((*stem)[3]);
  uint32_t t = cycles();
  g_l4.apply(l4, MODEL_STEM_OUTPUT0_NUMEL);
  g_l6.apply(l6, MODEL_STEM_OUTPUT1_NUMEL);
  g_y10.apply(y10, MODEL_STEM_OUTPUT2_NUMEL);
  g_qkv10.apply(qkv10, MODEL_STEM_OUTPUT3_NUMEL);
  g_requant_cycles += cycles() - t;
  t = cycles();
  int32_t err = attention_core(qkv10, kTokens10, {MODEL_MID_INPUT1_SCALE, MODEL_MID_INPUT1_ZERO_POINT}, g_o10,
                               {MODEL_MID_INPUT0_SCALE, MODEL_MID_INPUT0_ZERO_POINT});
  g_attention_cycles += cycles() - t;
  if (err != 0) return err;

  auto mid = g_module->execute(MODEL_MID_METHOD, {g_mid_in[0](g_o10), g_mid_in[1](qkv10), g_mid_in[2](y10),
                                                  g_mid_in[3](l4), g_mid_in[4](l6)});
  if (!mid.ok()) return static_cast<int32_t>(mid.error());
  if (mid->size() != MODEL_MID_NUM_OUTPUTS) return -1;
  int8_t* const qkv22 = data((*mid)[0]);
  int8_t* const z22 = data((*mid)[1]);
  int8_t* const y22 = data((*mid)[2]);
  int8_t* const box34 = data((*mid)[3]);
  int8_t* const cls34 = data((*mid)[4]);
  t = cycles();
  g_qkv22.apply(qkv22, MODEL_MID_OUTPUT0_NUMEL);
  g_z22.apply(z22, MODEL_MID_OUTPUT1_NUMEL);
  g_y22.apply(y22, MODEL_MID_OUTPUT2_NUMEL);
  g_box34.apply(box34, MODEL_MID_OUTPUT3_NUMEL);
  g_cls34.apply(cls34, MODEL_MID_OUTPUT4_NUMEL);
  g_requant_cycles += cycles() - t;
  t = cycles();
  err = attention_core(qkv22, kTokens22, {MODEL_HEAD_INPUT1_SCALE, MODEL_HEAD_INPUT1_ZERO_POINT}, g_o22,
                       {MODEL_HEAD_INPUT0_SCALE, MODEL_HEAD_INPUT0_ZERO_POINT});
  g_attention_cycles += cycles() - t;
  if (err != 0) return err;

  auto result = g_module->execute(MODEL_HEAD_METHOD, {g_head_in[0](g_o22), g_head_in[1](qkv22), g_head_in[2](z22),
                                                      g_head_in[3](y22), g_head_in[4](box34), g_head_in[5](cls34)});
  if (!result.ok()) return static_cast<int32_t>(result.error());
  if (result->size() != MODEL_HEAD_NUM_OUTPUTS) return -1;
  detector_attention_us = us(g_attention_cycles);
  detector_requant_us = us(g_requant_cycles);
#else
  auto result = g_module->execute(MODEL_DETECT_METHOD, inputs);
  if (!result.ok()) return static_cast<int32_t>(result.error());
  if (result->size() != MODEL_DETECT_NUM_OUTPUTS) return -1;
#endif
  out->count = decode((*result)[0].toTensor().const_data_ptr<int8_t>(), (*result)[1].toTensor().const_data_ptr<int8_t>(),
                      out->det);
  g_have_scores = true;
  out->total_us = us(cycles() - t0);
  out->npu_us = us(g_npu_cycles);
  for (uint32_t i = out->count; i < DETECTOR_MAX_DETECTIONS; ++i) out->det[i] = detection_t{};
  return 0;
}

extern "C" int32_t detector_last_scores(detector_scores_t* out) {
  if (!g_have_scores) return -1;
  out->score = g_best;
  out->anchors = kAnchors;
  out->scale = DETECTOR_CLS_SCALE;
  out->zero_point = DETECTOR_CLS_ZERO_POINT;
  out->threshold = kScoreThreshold;
  out->candidates = g_candidates;
  out->max = (g_score_max - DETECTOR_CLS_ZERO_POINT) * DETECTOR_CLS_SCALE;
  return 0;
}
