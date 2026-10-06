/*---------------------------------------------------------------------------
 * Copyright 2026 Arm Limited and/or its affiliates.
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
 *      Name:    retarget_stdio_log.c
 *      Purpose: Retarget stdio to a log buffer the debugger reads
 *
 * CMSIS-Compiler STDOUT/STDERR/STDIN:Custom backend for a board whose SW4
 * stays on SEUART, so that the Secure Enclave tools can program and reset it
 * without anyone touching the board: UART4, the usual console, is then not
 * connected to the host. Characters go into `console_log`, a ring buffer in
 * the DTCM, which a debugger reads while the core runs (tools/console_log.py);
 * the DTCM is not behind the data cache, so what is read is what was written.
 *---------------------------------------------------------------------------*/

#include <stdint.h>

#define CONSOLE_LOG_SIZE 0x8000U

/* `head` counts every character since start; text[head % size] is the next
   one to be written, so a reader sees the last `size` characters. */
struct console_log_s {
  uint32_t          magic; /* 'CLOG' */
  uint32_t          size;
  volatile uint32_t head;
  char              text[CONSOLE_LOG_SIZE];
} console_log __attribute__((section(".bss.console_log"), used));

/**
  Initialize stdio

  \return          0 on success, or -1 on error.
*/
int stdio_init (void) {
  console_log.magic = 0x474F4C43U;
  console_log.size  = CONSOLE_LOG_SIZE;
  console_log.head  = 0U;
  return 0;
}

static int log_putchar (int ch) {
  console_log.text[console_log.head % CONSOLE_LOG_SIZE] = (char)ch;
  console_log.head++;
  return ch;
}

/**
  Put a character to the stderr

  \param[in]   ch  Character to output
  \return          The character written, or -1 on write error.
*/
int stderr_putchar (int ch) {
  return log_putchar(ch);
}

/**
  Put a character to the stdout

  \param[in]   ch  Character to output
  \return          The character written, or -1 on write error.
*/
int stdout_putchar (int ch) {
  return log_putchar(ch);
}

/**
  Get a character from the stdio

  \return     The next character from the input, or -1 on read error.
*/
int stdin_getchar (void) {
  return -1;
}
