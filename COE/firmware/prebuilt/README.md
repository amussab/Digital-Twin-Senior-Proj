# Ready-to-flash node firmware

For flashing without installing the build tools. How to flash: [`../../ICS_START_HERE.md`](../../ICS_START_HERE.md)
(Parts 1 and 2).

| File | Board | Flash with | Address |
|---|---|---|---|
| `coe_node_stm32_LIVE.hex` | STM32 (NUCLEO-H755ZI-Q), both cores, LIVE mode (normal) | STM32CubeProgrammer | in the file (CM7 0x08000000, CM4 0x08100000) |
| `coe_node_stm32_LIVE_ICS_BACKUP.hex` | the same board, **backup**: features as the ICS 13 Nov training data (profile 4) | STM32CubeProgrammer | in the file |
| `coe_node_esp32.bin` | ESP32-C6 (nanoESP32-C6): bootloader + partition table + app in one file | esptool-js web page or `esptool` | **0x0** |

Built 16 Nov 2026 (afternoon): the node measures the machine speed from the Keyphasor (ADC
channel 7, no speed to set) and sends both proximity probes' values and its part of the end-to-end
latency with the 32 features (packet schema 3, 208 bytes). `coe_node_stm32_LIVE.hex`: bearing
profile 3 (RK4 S6700). `coe_node_stm32_LIVE_ICS_BACKUP.hex`: bearing profile 4, the ICS 13 Nov
training settings (single BSF, ± 0.08, 350 Hz envelope, B2X = copy of B1Y), see
[`../../ICS_UPDATE_16NOV.md`](../../ICS_UPDATE_16NOV.md). Flash one of the two; the same ESP32
file works with both. The ESP32 file has no hotspot settings: it runs only its own Wi-Fi
**COE-NODE**. SHA-256:

```
07ce47d27447628ad0881ab6c08cc1c77a68beead75ee07afe8ba9430b98d795  coe_node_stm32_LIVE.hex
7e4178f3f3e886e0d3bd3ccc45948c342c9ee165dfe67ae9d6812fc195da8ab0  coe_node_stm32_LIVE_ICS_BACKUP.hex
6e1cba1b38769de720110e0014b084f7bc95680391505f2d1c88b62d23a3d63d  coe_node_esp32.bin
```

Command-line flashing (instead of the apps in the guide):

```
STM32_Programmer_CLI -c port=SWD mode=UR reset=HWrst -d coe_node_stm32_LIVE.hex -v -hardRst
python -m pip install esptool
python -m esptool --chip esp32c6 -p COM6 -b 460800 write_flash 0x0 coe_node_esp32.bin
```
(`COM6`: use the ESP32's port from Device Manager.)

How these were made (COE): STM32 `tools\build.ps1 -Mode LIVE` (backup: `-Mode LIVE -IcsCompat`) in `stm32/`, then
`arm-none-eabi-objcopy -O ihex` of the CM4 and CM7 `.elf` files, joined into one file
(the CM4 file's end record removed). ESP32 `idf.py build` in `esp32/` without
`connection_settings.local.h`, then `esptool merge-bin -o coe_node_esp32.bin @flash_args`
in `esp32/build`.
