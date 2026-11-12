/** Exercise the actual JSON encoder with canaries and ASan/UBSan on the host. */
#include "feature_json.h"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    FeaturePacket packet = {.window_id = 3U, .rpm = 1750U,
                            .bearing_profile = 1U, .valid_channels = 15U};
    for (unsigned index = 0U; index < 32U; ++index)
        packet.features[index] = index % 2U == 0U ? FLT_MAX : FLT_MIN;
    unsigned char storage[FEATURE_JSON_BYTES + 32U];
    for (size_t capacity = 0U; capacity <= FEATURE_JSON_BYTES; ++capacity)
    {
        memset(storage, 0xCC, sizeof(storage));
        char *output = (char *)storage + 16U;
        const size_t length = feature_packet_to_json(&packet, "coe-memory-check", 42U, 1U,
                                                    output, capacity);
        for (size_t index = 0U; index < 16U; ++index)
        {
            assert(storage[index] == 0xCCU);
            assert(storage[16U + capacity + index] == 0xCCU);
        }
        if (length != 0U)
            assert(length < capacity && strlen(output) == length);
        else if (capacity != 0U)
            assert(output[0] == '\0');
    }
    puts("PASS JSON bounds: 2049 capacities, address/undefined-behavior sanitizers");
    return 0;
}
