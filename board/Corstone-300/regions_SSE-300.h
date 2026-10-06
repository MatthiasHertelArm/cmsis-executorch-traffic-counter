// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// Memory regions of the Corstone-300 FVP for the traffic counter, for the
// CMSIS-Toolbox linker script template (RTE/Device/SSE-300-MPS3/). All of
// the read-only part (code, the 2.3 MB program, the 0.5 MB test image) goes
// into __ROM0, all of the read-write part (the 1.25 MB NPU scratch, the two
// 0.5 MB input slots, the method pool, the thread stacks) into __RAM0: both
// in the secure alias of DDR4 block 1 (0x70000000, 256 MB), which the CPU and
// the NPU both reach. fvp_config.txt points the reset vector table there.
#ifndef REGIONS_SSE_300_H
#define REGIONS_SSE_300_H

//-------- <<< Use Configuration Wizard in Context Menu >>> --------------------
//------ With VS Code: Open Preview for Configuration Wizard -------------------

// <h> ROM Configuration
// =======================
// <h> __ROM0
//   <o> Base address <0x0-0xFFFFFFFF:8>
//   <i> Defines base address of memory region.
//   <i> Contains Startup and Vector Table
#define __ROM0_BASE   0x70000000
//   <o> Region size [bytes] <0x0-0xFFFFFFFF:8>
//   <i> Defines size of memory region.
#define __ROM0_SIZE   0x01000000
// </h>

// <h> __ROM1
#define __ROM1_BASE   0
#define __ROM1_SIZE   0
// </h>

// <h> __ROM2
#define __ROM2_BASE   0
#define __ROM2_SIZE   0
// </h>

// <h> __ROM3
#define __ROM3_BASE   0
#define __ROM3_SIZE   0
// </h>

// </h>

// <h> RAM Configuration
// =======================
// <h> __RAM0
//   <o> Base address <0x0-0xFFFFFFFF:8>
//   <i> Defines base address of memory region.
//   <i> Contains uninitialized RAM, Stack, and Heap
#define __RAM0_BASE   0x71000000
//   <o> Region size [bytes] <0x0-0xFFFFFFFF:8>
//   <i> Defines size of memory region.
#define __RAM0_SIZE   0x01000000
// </h>

// <h> __RAM1
#define __RAM1_BASE   0
#define __RAM1_SIZE   0
// </h>

// <h> __RAM2
#define __RAM2_BASE   0
#define __RAM2_SIZE   0
// </h>

// <h> __RAM3
#define __RAM3_BASE   0
#define __RAM3_SIZE   0
// </h>

// </h>

// <h> Stack / Heap Configuration
//   <o0> Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//   <o1> Heap Size (in Bytes) <0x0-0xFFFFFFFF:8>
#define __STACK_SIZE  0x00010000
#define __HEAP_SIZE   0x00018000
// </h>

#endif /* REGIONS_SSE_300_H */
