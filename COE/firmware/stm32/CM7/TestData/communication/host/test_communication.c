/** Host checks of the unchanged production sender/decoder with simulated HAL events.
 * Execute each sender scenario in its own process so static state starts at zero.
 */
#include "feature_uart.h"
#include "delivery_timing.h"
#include "latency_packet.h"
#include "feature_packet.h"
#include "communication_demo.h"
#include "final_payload_fixture.h"
#include "stm32h7xx_hal.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

HostSCB fake_scb;
UART_HandleTypeDef *fake_uart;
uint8_t *fake_buffer;
uint32_t fake_tick, fake_start_calls, fake_abort_calls, fake_cache_calls;
uint32_t fake_dma_irqs, fake_uart_irqs;
HAL_StatusTypeDef fake_start_status = HAL_OK, fake_init_status = HAL_OK;
int fake_pressed;
static unsigned replay_count;
void run_pipeline_batch_demo(void) { ++replay_count; }
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *handle);
void HAL_UART_ErrorCallback(UART_HandleTypeDef *handle);
void DMA1_Stream0_IRQHandler(void);
void USART2_IRQHandler(void);

/** Execute the same 22 decoder tests compiled into the MCU plus button edge checks. */
static void test_decoder(void)
{
    run_communication_demo();
    assert(communication_checks_passed && communication_checked_cases == 22U);
    assert(communication_failed_check == 0U);
    feature_uart_ready = true;
    poll_communication_replay();
    fake_pressed = 1;
    fake_tick = 100U;
    poll_communication_replay();
    fake_tick = 120U;
    poll_communication_replay();
    assert(replay_count == 0U); /* Not stable long enough. */
    fake_tick = 130U;
    poll_communication_replay();
    assert(replay_count == 1U);
    fake_tick = 1000U;
    poll_communication_replay();
    assert(replay_count == 1U); /* Holding does not repeat. */
    fake_pressed = 0;
    poll_communication_replay();
    fake_tick = 1030U;
    poll_communication_replay();
    fake_pressed = 1;
    fake_tick = 1100U;
    poll_communication_replay();
    fake_tick = 1130U;
    poll_communication_replay();
    assert(replay_count == 2U);
}

/** Verify owned copies, two-slot backpressure, order, IRQ forwarding and delayed service. */
static void test_queue(void)
{
    uint8_t first[160], second[160], third[160];
    memset(first, 11, sizeof(first));
    memset(second, 22, sizeof(second));
    memset(third, 33, sizeof(third));
    assert(!feature_uart_send(first)); /* Not initialized. */
    assert(feature_uart_init());
    assert(!feature_uart_send(NULL));
    fake_scb.CCR = SCB_CCR_DC_Msk;
    assert(feature_uart_send(first));
    assert(fake_start_calls == 1U && fake_buffer[0] == 11U && fake_buffer[159] == 11U);
    assert(((uintptr_t)fake_buffer % 32U) == 0U && fake_cache_calls == 1U);
    memset(first, 77, sizeof(first));
    assert(fake_buffer[0] == 11U); /* Caller storage may immediately change. */
    assert(feature_uart_send(second));
    memset(second, 88, sizeof(second));
    assert(!feature_uart_send(third));
    assert(feature_uart_queue_full_count == 1U && fake_start_calls == 1U);
    DMA1_Stream0_IRQHandler();
    USART2_IRQHandler();
    assert(fake_dma_irqs == 1U && fake_uart_irqs == 1U);
    UART_HandleTypeDef other = {0};
    HAL_UART_TxCpltCallback(&other);
    HAL_UART_ErrorCallback(&other);
    feature_uart_process();
    assert(feature_uart_packets_sent == 0U && feature_uart_ready);
    HAL_UART_TxCpltCallback(fake_uart);
    fake_tick = 1000U; /* Completed transfer must not time out due to delayed service. */
    feature_uart_process();
    assert(feature_uart_packets_sent == 1U && fake_start_calls == 2U);
    assert(fake_buffer[0] == 22U && fake_buffer[159] == 22U && fake_cache_calls == 2U);
    HAL_UART_TxCpltCallback(fake_uart);
    feature_uart_process();
    assert(feature_uart_packets_sent == 2U && feature_uart_errors == 0U);
    assert(feature_uart_send(third));
    assert(fake_buffer[0] == 33U && fake_start_calls == 3U);
}

/** Measure at UART completion, including DWT wrap; main polling adds no time. */
static void test_uart_timing(void)
{
    uint8_t bytes[160] = {0};
    fake_tick = 10730U; /* The modeled 400-MHz cycle counter wraps at about 10737 ms. */
    assert(feature_uart_init() && feature_uart_timer_ready);
    assert(feature_uart_send(bytes));
    fake_tick += 14U;
    HAL_UART_TxCpltCallback(fake_uart);
    assert(feature_uart_last_tx_us == 14000.0f);
    assert(feature_uart_max_tx_us == 14000.0f);
    fake_tick += 30U;
    feature_uart_process();
    assert(feature_uart_packets_sent == 1U);
    assert(feature_uart_last_tx_us == 14000.0f);
}

/** Check timeout across the 32-bit tick wrap and a latched stop after failure. */
static void test_timeout(void)
{
    uint8_t bytes[160] = {0};
    fake_tick = UINT32_MAX - 15U;
    assert(feature_uart_init() && feature_uart_send(bytes));
    fake_tick += 101U;
    feature_uart_process();
    assert(!feature_uart_ready && feature_uart_errors == 1U && fake_abort_calls == 1U);
    assert(!feature_uart_send(bytes) && fake_start_calls == 1U);
}

/** Reject a HAL start failure; never report that packet as transmitted. */
static void test_start_failure(void)
{
    uint8_t bytes[160] = {0};
    assert(feature_uart_init());
    fake_start_status = HAL_BUSY;
    assert(!feature_uart_send(bytes));
    assert(!feature_uart_ready && feature_uart_errors == 1U && feature_uart_packets_sent == 0U);
}

/** Defer UART error handling until main services the event. */
static void test_uart_error(void)
{
    uint8_t bytes[160] = {0};
    assert(feature_uart_init() && feature_uart_send(bytes));
    HAL_UART_ErrorCallback(fake_uart);
    assert(feature_uart_ready);
    feature_uart_process();
    assert(!feature_uart_ready && feature_uart_errors == 1U && fake_abort_calls == 1U);
}

/** Count accepted frames while checking every recovered value against the fixture. */
static unsigned feed_checked(FeaturePacketStream *stream, const uint8_t *bytes, size_t length)
{
    unsigned accepted = 0U;
    FeaturePacket packet;
    for (size_t index = 0U; index < length; ++index)
        if (feature_packet_push(stream, bytes[index], &packet))
        {
            assert(packet.window_id == 17U && packet.rpm == 1750U
                   && packet.bearing_profile == 1U && packet.valid_channels == 15U);
            for (size_t value = 0U; value < 32U; ++value)
                assert(packet.features[value] == (float)((value / 8U) * 10U + value % 8U + 1U));
            ++accepted;
        }
    return accepted;
}

/** Exhaust all partial lengths and single-byte insertions/deletions: 480 stream cases. */
static void test_resynchronization(void)
{
    for (size_t cut = 1U; cut < 160U; ++cut)
    {
        FeaturePacketStream stream = {0};
        assert(feed_checked(&stream, final_payload_expected, cut) == 0U);
        assert(feed_checked(&stream, final_payload_expected, 160U) == 1U);
    }
    for (size_t removed = 0U; removed < 160U; ++removed)
    {
        FeaturePacketStream stream = {0};
        assert(feed_checked(&stream, final_payload_expected, removed) == 0U);
        assert(feed_checked(&stream, final_payload_expected + removed + 1U, 159U - removed) == 0U);
        assert(feed_checked(&stream, final_payload_expected, 160U) == 1U);
    }
    for (size_t added = 0U; added <= 160U; ++added)
    {
        FeaturePacketStream stream = {0};
        const uint8_t extra = 0x5AU;
        unsigned accepted = feed_checked(&stream, final_payload_expected, added);
        accepted += feed_checked(&stream, &extra, 1U);
        accepted += feed_checked(&stream, final_payload_expected + added, 160U - added);
        accepted += feed_checked(&stream, final_payload_expected, 160U);
        assert(accepted == ((added == 0U || added == 160U) ? 2U : 1U));
    }
}


/** Verify matched ACKs, stale/corrupt/wrong metadata rejection, wall-clock wrap and timeouts. */
static void test_delivery(void)
{
    assert(feature_uart_init());
    delivery_timing_start_batch();
    fake_tick = UINT32_MAX - 20U;
    delivery_timing_begin();
    delivery_timing_bind_packet(final_payload_expected);
    LatencyAck ack = {1U, 17U, 1750U, 1U, 7000U, 1000U, 6000U};
    uint8_t bytes[LATENCY_ACK_BYTES];
    latency_ack_encode(&ack, bytes);
    bytes[25] ^= 1U;
    for (unsigned i = 0; i < sizeof(bytes); ++i) delivery_timing_receive_byte(bytes[i]);
    assert(delivery_timing_completed == 0U);
    ack.measurement_id = 99U; /* Late/unrelated ACK cannot complete a window. */
    latency_ack_encode(&ack, bytes);
    for (unsigned i = 0; i < sizeof(bytes); ++i) delivery_timing_receive_byte(bytes[i]);
    ack.measurement_id = 1U;
    ack.rpm = 3600U; /* Correct ID but wrong vector also cannot complete it. */
    latency_ack_encode(&ack, bytes);
    for (unsigned i = 0; i < sizeof(bytes); ++i) delivery_timing_receive_byte(bytes[i]);
    ack.rpm = 1750U;
    latency_ack_encode(&ack, bytes);
    fake_tick += 100U; /* Cross the 32-bit millisecond wrap. */
    for (unsigned i = 0; i < sizeof(bytes); ++i) delivery_timing_receive_byte(bytes[i]);
    fake_tick += 30U; /* Main polling delay must NOT inflate IRQ-recorded total. */
    ++feature_uart_packets_sent;
    feature_uart_last_tx_us = 3500.0f;
    delivery_timing_finish(90.0f);
    assert(delivery_timing_completed == 1U && delivery_timing_timeouts == 0U);
    assert(delivery_timing_results[0].valid);
    assert(delivery_timing_results[0].scope_roundtrip_ms == 100.0f);
    assert(delivery_timing_results[0].processing_ms == 90.0f);
    assert(delivery_timing_results[0].uart_tx_ms == 3.5f);
    assert(delivery_timing_results[0].esp_queue_ms == 1.0f);
    assert(delivery_timing_results[0].esp_network_ms == 6.0f);
    assert(!delivery_timing_batch_complete);
    delivery_timing_begin();
    delivery_timing_bind_packet(final_payload_expected);
    /* Duplicate previous ACK must not validate a new timing ID. */
    for (unsigned i = 0; i < sizeof(bytes); ++i) delivery_timing_receive_byte(bytes[i]);
    ++feature_uart_packets_sent;
    feature_uart_last_tx_us = 3600.0f;
    delivery_timing_finish(90.0f);
    assert(delivery_timing_results[1].processing_ms == 90.0f);
    assert(delivery_timing_results[1].uart_tx_valid);
    assert(delivery_timing_results[1].uart_tx_ms == 3.6f);
    assert(delivery_timing_results[1].scope_roundtrip_ms == 0.0f);
    assert(delivery_timing_timeouts == 1U && delivery_timing_completed == 1U);
    assert(!delivery_timing_results[1].valid);
    assert(!delivery_timing_batch_complete);
    printf("Delivery checks: matched/stale/corrupt/wrong vector/timeout/tick-wrap/IRQ timestamp passed\n");
}

/** Run one isolated scenario selected by the verification script. */
int main(int argc, char **argv)
{
    assert(argc == 2);
    if (strcmp(argv[1], "decoder") == 0) test_decoder();
    else if (strcmp(argv[1], "delivery") == 0) test_delivery();
    else if (strcmp(argv[1], "uart_timing") == 0) test_uart_timing();
    else if (strcmp(argv[1], "queue") == 0) test_queue();
    else if (strcmp(argv[1], "timeout") == 0) test_timeout();
    else if (strcmp(argv[1], "start_failure") == 0) test_start_failure();
    else if (strcmp(argv[1], "uart_error") == 0) test_uart_error();
    else if (strcmp(argv[1], "resynchronization") == 0) test_resynchronization();
    else if (strcmp(argv[1], "init_failure") == 0)
    {
        fake_init_status = HAL_ERROR;
        assert(!feature_uart_init() && !feature_uart_ready && fake_start_calls == 0U);
    }
    else assert(0);
    printf("PASS %s\n", argv[1]);
    return 0;
}
