/*---------------------------------------------------------------------------
 * Copyright (c) 2025-2026 Arm Limited (or its affiliates). All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * AppKit-E7 (M55_HP) board bring-up for the traffic counter. Derived from the
 * Ensemble pack's board layer main.c (the DevKit-E8's, which the AppKit-E7
 * layer follows): the Secure Enclave link, the clocks, the MIPI DPHY power for
 * the panel and the camera, the USB PHY for SDS, then the application in an
 * RTX thread.
 *---------------------------------------------------------------------------*/

#include "RTE_Components.h"
#include CMSIS_device_header

#include "board_config.h"
#include "main.h"

#include "se_services_port.h"
#include "board_display.h"
#ifdef RTE_CMSIS_RTOS2
#include "cmsis_os2.h"
#endif

/* VBAT power control bits for the MIPI TX DPHY and its PLL (pack layer main.c) */
#define VBAT_PWR_CTRL_TX_DPHY_PWR_MASK        (1U <<  0) /* Mask off the power supply for MIPI TX DPHY */
#define VBAT_PWR_CTRL_TX_DPHY_ISO             (1U <<  1) /* Enable isolation for MIPI TX DPHY */
#define VBAT_PWR_CTRL_RX_DPHY_PWR_MASK        (1U <<  4) /* Mask off the power supply for MIPI RX DPHY */
#define VBAT_PWR_CTRL_RX_DPHY_ISO             (1U <<  5) /* Enable isolation for MIPI RX DPHY */
#define VBAT_PWR_CTRL_DPHY_PLL_PWR_MASK       (1U <<  8) /* Mask off the power supply for MIPI PLL */
#define VBAT_PWR_CTRL_DPHY_PLL_ISO            (1U <<  9) /* Enable isolation for MIPI PLL */
#define VBAT_PWR_CTRL_DPHY_VPH_1P8_PWR_BYP_EN (1U << 12) /* dphy vph 1p8 power bypass enable */

/*
  Power up the MIPI DPHY (the display's physical layer), as the pack's
  DevKit-e8 layer does in its vbat_init().
*/
static void dphy_power_init(void)
{
    VBAT->PWR_CTRL &= ~(VBAT_PWR_CTRL_TX_DPHY_PWR_MASK | VBAT_PWR_CTRL_RX_DPHY_PWR_MASK |
                        VBAT_PWR_CTRL_DPHY_PLL_PWR_MASK | VBAT_PWR_CTRL_DPHY_VPH_1P8_PWR_BYP_EN);
    VBAT->PWR_CTRL &= ~(VBAT_PWR_CTRL_TX_DPHY_ISO | VBAT_PWR_CTRL_RX_DPHY_ISO | VBAT_PWR_CTRL_DPHY_PLL_ISO);
}

#ifdef RTE_SDS_IO_CLIENT_USB
#include "services_lib_api.h"
#include "aipm.h"
static void usb_power_init(void)
{
    uint32_t error_code = 0;
    run_profile_t runp;
    if (SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_CLK_20M, true, &error_code) != 0 || error_code != 0) {
        return;
    }
    if (SERVICES_get_run_cfg(se_services_s_handle, &runp, &error_code) != 0) {
        return;
    }
    runp.phy_pwr_gating |= USB_PHY_MASK;
    SERVICES_set_run_cfg(se_services_s_handle, &runp, &error_code);
}
#endif

#ifdef RTE_CMSIS_RTOS2
#ifndef APP_THREAD_STACK_SIZE
#define APP_THREAD_STACK_SIZE 0x8000
#endif
static uint64_t app_thread_stack[APP_THREAD_STACK_SIZE / 8] __attribute__((section(".bss.ai_pool")));
static const osThreadAttr_t app_thread_attr = {
    .name       = "app",
    .stack_mem  = app_thread_stack,
    .stack_size = sizeof(app_thread_stack),
    .priority   = osPriorityNormal,
};

static void app_thread(void *argument)
{
    (void)argument;
    app_main();
    for (;;) {
        osDelay(osWaitForever);
    }
}
#endif

int main(void)
{
    /* Apply the Conductor pin configuration (includes the UART4 console pins) */
      board_pins_config();

    /* Apply the Conductor GPIO configuration */
    board_gpios_config();

    /* Bring up the Secure Enclave services (MHU link to the SE) */
    se_services_port_init();

    /* Request the clocks the SE has to enable for this core */
    board_clocks_config(CLKEN_HFOSC_MASK | CLKEN_CLK_100M_MASK);

    /* Power up the MIPI DPHY before the display driver touches it */
    dphy_power_init();

#ifdef RTE_SDS_IO_CLIENT_USB
    /* The USB device (SDS over the User USB): its 20 MHz clock and the PHY's
       power, through the Secure Enclave, as the pack's AppKit-E7 layer does. */
    usb_power_init();
#endif

    /* Initialize STDIO (UART4 on the PRG USB connector) */
    stdio_init();

    #if defined(ETHOSU_ARCH)
    /* Initialize Ethos NPU */
    ethos_setup();
    #endif

#ifdef RTE_CMSIS_RTOS2
    /* The application's threads and the SDS Stream component need
       CMSIS-RTOS2: run the application in an RTX thread. Its stack lives in
       the bulk SRAM next to the ExecuTorch pools. */
    osKernelInitialize();
    osThreadNew(app_thread, NULL, &app_thread_attr);
    osKernelStart();
    for (;;) {}
#else
    return app_main();
#endif
}
