# CM7 firmware mode. Edit this one line, then build as usual.
#   REPLAY    recorded-data DSP/UART checks at startup (default)
#   LIVE_RAW  AD7606 acquisition bring-up: raw counts and status on the
#             ST-LINK virtual COM port; no DSP and no feature packets
#   LIVE      AD7606 -> DSP -> 32 features -> ESP32; settings and calibration
#             in App/config/live_config.h (packets only once calibrated)
set(COE_FIRMWARE_MODE REPLAY)
