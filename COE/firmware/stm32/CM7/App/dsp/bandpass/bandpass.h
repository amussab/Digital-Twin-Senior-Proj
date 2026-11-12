/**
 * @file bandpass.h
 * Public interface for one channel's persistent CMSIS-DSP band-pass filter.
 * Call reset once at stream startup, then process chronological sample blocks.
 */
#ifndef COE_BANDPASS_H
#define COE_BANDPASS_H

#include <stdbool.h>
#include <stddef.h>

#define BANDPASS_SECTION_COUNT 4U

/** One state per channel; all channels share the read-only coefficients. */
typedef struct
{
    float delay[2U * BANDPASS_SECTION_COUNT]; /* CMSIS d1,d2 for each section. */
    bool valid;                             /* False after a processing error. */
} BandpassState;

/**
 * Clear the delays and start a new independent channel stream.
 * A NULL state is ignored. Do not reset at block or analysis-window boundaries.
 */
void bandpass_reset(BandpassState *state);

/**
 * Filter count samples from input into output using one channel's state.
 * Both buffers contain at least count floats, with matching acceleration units.
 * Exactly in-place filtering is allowed; partially overlapping buffers are not.
 * A zero-length call succeeds for a valid state, even with NULL buffers.
 *
 * Return true on success. Invalid buffers, nonfinite data or numerical overflow
 * return false and invalidate the state. Discard that output; reset to recover.
 * The function retains delays but never retains pointers to the sample buffers.
 */
bool bandpass_process(BandpassState *state, const float *input,
                      float *output, size_t count);

#endif
