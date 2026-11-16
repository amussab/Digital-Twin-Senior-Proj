# Ready-to-flash node firmware

For flashing without installing the build tools. How to flash: [`../../ICS_START_HERE.md`](../../ICS_START_HERE.md)
(Parts 1 and 2).

| File | Board | Flash with | Address |
|---|---|---|---|
| `coe_node_stm32_LIVE.hex` | STM32 (NUCLEO-H755ZI-Q), both cores, LIVE mode | STM32CubeProgrammer | in the file (CM7 0x08000000, CM4 0x08100000) |
| `coe_node_esp32.bin` | ESP32-C6 (nanoESP32-C6): bootloader + partition table + app in one file | esptool-js web page or `esptool` | **0x0** |

Built 16 Nov 2026 (afternoon): the node measures the machine speed from the Keyphasor (ADC
channel 7, no speed to set) and sends both proximity probes' values and its part of the end-to-end
latency, the ADC sample rate, the turns and speed range per window and its health counters with the
32 features (packet schema 3, 232 bytes); bearing profile 3 (RK4 S6700). What the AI host gets:
[`../../ICS_UPDATE_16NOV.md`](../../ICS_UPDATE_16NOV.md). The ESP32 file has no hotspot settings: it runs only its own Wi-Fi
**COE-NODE**. SHA-256:

```
9441d943f19f460c99596a9f42481bff104c55b2326693b5f21b740538c32f7c  coe_node_stm32_LIVE.hex
328004097aee7f0890cdd29a6eb397b4bf2e7398689fb75f6db9af4dd089ea4e  coe_node_esp32.bin
```

Command-line flashing (instead of the apps in the guide):

```
STM32_Programmer_CLI -c port=SWD mode=UR reset=HWrst -d coe_node_stm32_LIVE.hex -v -hardRst
python -m pip install esptool
python -m esptool --chip esp32c6 -p COM6 -b 460800 write_flash 0x0 coe_node_esp32.bin
```
(`COM6`: use the ESP32's port from Device Manager.)

How these were made (COE): STM32 `tools\build.ps1 -Mode LIVE` in `stm32/`, then
`arm-none-eabi-objcopy -O ihex` of the CM4 and CM7 `.elf` files, joined into one file
(the CM4 file's end record removed). ESP32 `idf.py build` in `esp32/` without
`connection_settings.local.h`, then `esptool merge-bin -o coe_node_esp32.bin @flash_args`
in `esp32/build`.
