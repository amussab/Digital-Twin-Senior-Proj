/**
 * @file envelope.h
 * Convert one channel's signed band-pass blocks into a 5-kSamples/s envelope.
 * The path is absolute value, causal low-pass filtering, then one-in-six selection.
 */
#ifndef COE_ENVELOPE_H
#define COE_ENVELOPE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define ENVELOPE_SECTION_COUNT 4U

/** One persistent state per channel; reset only when starting a new stream. */
typedef struct
{
    float delay[2U * ENVELOPE_SECTION_COUNT]; /* CMSIS d1,d2 for each section. */
    uint32_t phase;                          /* Input samples processed modulo six. */
    bool valid;                             /* False after an error until reset. */
} EnvelopeState;

/** Clear filter delays and set the selection phase to zero. NULL is ignored. */
void envelope_reset(EnvelopeState *state);

/**
 * Make a reduced envelope from count chronological signed band-pass samples.
 * input and output each provide at least count floats: output is also the
 * full-rate filtering workspace, then its front is compacted in place.
 * On success, only the first *output_count values are retained envelope samples.
 * Keep one state per channel across blocks AND analysis-window boundaries.
 *
 * Input/output units agree. No averaging, amplitude correction or clipping is
 * applied. Exactly in-place processing is allowed; partial overlap is unsupported.
 * output_count must be non-NULL. A zero count succeeds for a valid state, sets
 * *output_count to zero and permits NULL input/output without advancing state.
 * Invalid arguments/data or nonfinite arithmetic return false, invalidate the
 * state and leave *output_count zero. Discard output and reset before recovery.
 * The caller owns the buffers; this function allocates and retains no samples.
 */
bool envelope_process(EnvelopeState *state, const float *input, float *output,
                       size_t count, size_t *output_count);

#endif
