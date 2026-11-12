/**
 * @file pipeline.h
 * Connect all four-channel DSP stages and publish a fixed 32-feature AI packet.
 */
#ifndef COE_PIPELINE_H
#define COE_PIPELINE_H

#include "pipeline_config.h"
#include "bandpass.h"
#include "time_features.h"
#include "envelope.h"
#include "envelope_spectrum.h"
#include "feature_payload.h"

/** One completed window: time/frequency features, encoded AI packet and diagnostic envelopes. */
typedef struct
{
    uint32_t window_id;       /* Starts at one for each initialized stream. */
    uint32_t nominal_rpm;
    uint32_t sample_count;    /* Original samples/channel, rounded to nearest sample. */
    TimeDomainFeatures features[COE_CHANNEL_COUNT];
    FrequencyDomainFeatures frequency_features[COE_CHANNEL_COUNT];
    bool valid[COE_CHANNEL_COUNT]; /* Whether all three time features are defined. */
    AiFeaturePayload ai_payload; /* Owned values/bytes within this borrowed result structure. */
    uint32_t envelope_sample_count; /* Common retained count for all channels. */
    uint32_t envelope_sample_rate_hz;
    uint32_t envelope_first_input_offset; /* First retained index within this window. */
    const float *envelope[COE_CHANNEL_COUNT]; /* Borrowed arrays, each containing count samples. */
} PipelineWindowResult;

/**
 * Receive each completed window synchronously. The result and its envelope
 * arrays are borrowed: consume/copy them during this call if retaining them.
 * Later input overwrites the same envelope storage. Do not reenter the pipeline.
 * Return false if the result cannot be accepted; keep this callback short.
 */
typedef bool (*PipelineWindowCallback)(const PipelineWindowResult *result,
                                       void *context);

/** Fixed storage for four envelope records; keep it alive for the whole stream. */
typedef struct
{
    float samples[COE_CHANNEL_COUNT][COE_MAX_ENVELOPE_SAMPLES];
} PipelineEnvelopeBuffer;

/** Small persistent channel state; the large envelope buffer is supplied separately. */
typedef struct
{
    BandpassState filters[COE_CHANNEL_COUNT];
    TimeFeatureState statistics[COE_CHANNEL_COUNT];
    EnvelopeState envelopes[COE_CHANNEL_COUNT];
    float filtered_block[COE_DMA_BLOCK_SAMPLES]; /* Shared between channel calls. */
    PipelineEnvelopeBuffer *envelope_buffer;
    EnvelopeSpectrumWorkspace *spectrum_workspace; /* Shared only between sequential stream calls. */
    const BearingFeatureProfile *bearing_profile;
    uint32_t nominal_rpm;
    uint32_t window_revolutions; /* 20, or fewer at slow speeds (see pipeline_config.h). */
    uint32_t window_samples;
    uint32_t samples_in_window;
    uint32_t envelope_samples_in_window;
    uint32_t envelope_start_phase; /* Input position modulo six at this window's start. */
    uint32_t completed_windows;
    PipelineWindowCallback on_window;
    void *callback_context;
    bool valid;
} PipelineState;

/**
 * Start a stream at a settled speed of COE_MIN_RPM..COE_MAX_RPM; reset all states.
 * Call once at stream startup, not per window. False rejects invalid arguments
 * or a window too large for the fixed buffer. Provide separate buffer storage
 * for each stream; it must remain alive and writable until the stream ends.
 * Contents are written before reading and need no clearing at startup/windows.
 * Keep the approximately 55-KB envelope buffer and 63-KB FFT workspace in static
 * SRAM, not the stack. Pass an explicit bearing profile, valid at this RPM.
 * Input acceleration is calibrated in g; the packet declares this unit.
 */
bool pipeline_init(PipelineState *pipeline, uint32_t nominal_rpm,
                    PipelineEnvelopeBuffer *envelope_buffer,
                    EnvelopeSpectrumWorkspace *spectrum_workspace,
                    const BearingFeatureProfile *bearing_profile,
                    PipelineWindowCallback callback, void *context);

/**
 * Consume four chronological channel arrays, each containing count acceleration samples in g.
 * Order: bearing 1 X, bearing 1 Y, bearing 2 X, bearing 2 Y. Each call internally
 * splits at scratch/window boundaries. Signed time features are updated before
 * envelope processing overwrites the scratch block. Input is not modified.
 * Call from one context outside DMA interrupts. A valid zero count is a no-op.
 * False invalidates the stream until reinitialized; discard the incomplete window.
 * Previously accepted complete windows remain valid. No heap allocation is used.
 */
bool pipeline_process_block(PipelineState *pipeline,
                             const float *const channels[COE_CHANNEL_COUNT],
                             size_t count);

#endif
