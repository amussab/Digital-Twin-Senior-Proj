/**
 * @file firmware_mode.h
 * Build-time firmware mode, selected in CM7/firmware_mode.cmake.
 */
#ifndef COE_FIRMWARE_MODE_H
#define COE_FIRMWARE_MODE_H

/* Values must match CM7/CMakeLists.txt. */
#define COE_MODE_REPLAY 0   /* Recorded-data checks; default. */
#define COE_MODE_LIVE_RAW 1 /* AD7606 bring-up: raw counts, no DSP/packets. */
#define COE_MODE_LIVE 2     /* AD7606 -> DSP -> packets to the ESP32. */

#ifndef COE_FIRMWARE_MODE
#error "COE_FIRMWARE_MODE is set by CM7/CMakeLists.txt"
#endif

#endif
