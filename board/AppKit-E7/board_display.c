/*---------------------------------------------------------------------------
 * Copyright (c) 2026 Arm Limited (or its affiliates). All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AppKit-E7 display bring-up through the pack's CDC200 driver, the sequence
 * the pack's vStream VideoOut driver uses: Initialize, PowerControl, enable
 * the start-of-frame event, configure the display (which also initialises
 * the MIPI DSI link and the ILI9806E panel), set a frame buffer, Start.
 *---------------------------------------------------------------------------*/

#include "RTE_Components.h"
#include CMSIS_device_header

#include "Driver_CDC200.h"
#include "cdc.h"
#include "board_display.h"

#include "RTE_Components.h"
#ifdef RTE_CMSIS_RTOS2
#include "cmsis_os2.h"
#endif

extern ARM_DRIVER_CDC200 Driver_CDC200;
static ARM_DRIVER_CDC200 *cdc = &Driver_CDC200;

static volatile uint32_t frames_started;

static void cdc_callback(uint32_t event)
{
    if (event & ARM_CDC_SCANLINE0_EVENT) {
        frames_started++;
    }
}

int32_t display_init(void)
{
    int32_t status;

    status = cdc->Initialize(cdc_callback);
    if (status != ARM_DRIVER_OK) {
        return status;
    }
    status = cdc->PowerControl(ARM_POWER_FULL);
    if (status != ARM_DRIVER_OK) {
        return status;
    }
    status = cdc->Control(CDC200_SCANLINE0_EVENT, 1U);
    if (status != ARM_DRIVER_OK) {
        return status;
    }
    return cdc->Control(CDC200_CONFIGURE_DISPLAY, 0U);
}

int32_t display_start(const void *fb)
{
    int32_t status = cdc->Control(CDC200_FRAMEBUF_UPDATE, (uint32_t)fb);
    if (status != ARM_DRIVER_OK) {
        return status;
    }
    return cdc->Start();
}

int32_t display_present(const void *fb)
{
    return cdc->Control(CDC200_FRAMEBUF_UPDATE_VSYNC, (uint32_t)fb);
}

uint32_t display_frame_count(void)
{
    return frames_started;
}

int32_t display_wait_frame(uint32_t count)
{
#ifdef RTE_CMSIS_RTOS2
    /* Under the RTOS the wait gives the CPU away (another thread renders the
       next frame meanwhile): poll once per tick, give up after four frames. */
    if (osKernelGetState() == osKernelRunning) {
        for (uint32_t ticks = 0; ticks < 68U; ++ticks) {
            if ((int32_t)(frames_started - count) > 0) {
                return 0;
            }
            osDelay(1U);
        }
        return -1;
    }
#endif
    /* A 60 Hz frame is 16.7 ms; give up after roughly four of them. */
    for (uint32_t spins = 0; spins < 4000000U; ++spins) {
        if ((int32_t)(frames_started - count) > 0) {
            return 0;
        }
    }
    return -1;
}

int32_t display_wait_shown(const void *fb)
{
    /* The layer's frame buffer address is a shadowed register: it reads back
       the working value, so it shows `fb` only once the reload at the start of
       the vertical blanking has happened (hardware reference manual, "Shadowed
       Registers Reload"). From then on the buffer shown before is not read any
       more. A scanline-0 count cannot tell that: a request that arrives in the
       blanking interval, after the reload point, takes one frame longer.
       `fb` is in the bulk SRAM, where local and global addresses are equal. */
    const CDC_Type *regs = (const CDC_Type *)CDC_BASE;

    for (uint32_t ticks = 0; ticks < 68U; ++ticks) {
        if (regs->CDC_LAYER_CFG[CDC_LAYER_1].CDC_L_CFB_ADDR == (uint32_t)fb) {
            return 0;
        }
#ifdef RTE_CMSIS_RTOS2
        if (osKernelGetState() == osKernelRunning) {
            osDelay(1U);
            continue;
        }
#endif
        for (volatile uint32_t spin = 0; spin < 60000U; ++spin) {
        }
    }
    return -1;
}
