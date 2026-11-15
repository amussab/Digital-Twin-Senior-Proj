/**
 * @file live_stream.h
 * LIVE mode: AD7606 blocks -> scaling -> pipeline -> 204-byte packets (schema 2)
 * on USART2 to the ESP32. DSP runs in the main loop, never in an interrupt.
 * The speed comes from the RK4 Keyphasor (AIN7): windows start once it is
 * steady, follow it, and stop when it is lost. The packet also carries the
 * measured speed and the RK4 PROX 1 / PROX 3 values for the same samples.
 * Any acquisition gap discards the window in progress and restarts the
 * pipeline, so a window never spans missing samples.
 */
#ifndef COE_LIVE_STREAM_H
#define COE_LIVE_STREAM_H

#include <stdbool.h>
#include <stdint.h>
#include "pipeline_config.h"

/* Watch variables (counters restart at each B1 start). */
extern volatile bool live_calibrated;          /* All four volts-per-g values set. */
extern volatile uint32_t live_windows;         /* Completed windows. */
extern volatile uint32_t live_packets_queued;  /* Accepted by feature_uart_send. */
extern volatile uint32_t live_packets_withheld; /* Not sent: uncalibrated. */
extern volatile uint32_t live_packet_failures; /* Refused by the UART queue. */
extern volatile uint32_t live_discarded_blocks; /* Dropped around acquisition gaps. */
extern volatile uint32_t live_restarts;        /* Pipeline restarts after gaps/errors. */
extern volatile uint32_t live_pipeline_errors;
extern volatile float live_max_block_ms;       /* Longest pipeline_process_block call. */
extern volatile float live_dsp_load_percent;   /* DSP time / real time, latest second. */
extern volatile float live_last_rms[COE_CHANNEL_COUNT]; /* Band-pass RMS, latest window. */
/* S2: ms from the newest sample of the window's last block to features ready. */
extern volatile float live_last_window_latency_ms;
extern volatile float live_max_window_latency_ms;
extern volatile uint32_t live_rpm;             /* Whole-RPM speed of the windows (Keyphasor); 0 = none. */
extern volatile float live_measured_rpm;       /* Keyphasor speed, last 4 turns; 0 = not steady/none. */
extern volatile uint32_t live_speed_changes;   /* Window restarts at a new measured speed. */
extern volatile uint32_t live_speed_losses;    /* Keyphasor speed lost while windows ran. */
extern volatile uint32_t live_unsteady_windows; /* Not sent: speed moved during the window. */
extern volatile float live_last_gap_um[2];     /* PROX 1, PROX 3, latest window (0 = not valid). */
extern volatile float live_last_x1_um_pp[2];
extern volatile float live_last_x1_phase_deg[2];

/** Initialize the pipeline and print the banner. feature_uart_init() first. */
bool live_stream_init(void);

/** Call continuously from the main loop. */
void live_stream_poll(void);

#endif
