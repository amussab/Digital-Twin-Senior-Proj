# Ready-to-flash node firmware

For flashing without installing the build tools. How to flash: [`../../ICS_START_HERE.md`](../../ICS_START_HERE.md)
(Parts 1 and 2).

| File | Board | Flash with | Address |
|---|---|---|---|
| `coe_node_stm32_LIVE.hex` | STM32 (NUCLEO-H755ZI-Q), both cores, LIVE mode | STM32CubeProgrammer | in the file (CM7 0x08000000, CM4 0x08100000) |
| `coe_node_esp32.bin` | ESP32-C6 (nanoESP32-C6): bootloader + partition table + app in one file | esptool-js web page or `esptool` | **0x0** |

Built 16 Nov 2026 from `COE/firmware` at commit `bce7dcc`: the node measures the machine speed
from the Keyphasor (ADC channel 7, no speed to set) and sends both proximity probes' values with
the 32 features (packet schema 2, 204 bytes); bearing profile 3 (RK4 S6700). The ESP32 file has
no hotspot settings: it runs only its own Wi-Fi **COE-NODE**. SHA-256:

```
7d4a019ea3d204d1fa266a862c92a66b40b2b493f725356a0e53accf01e5418e  coe_node_stm32_LIVE.hex
1ff562c54035b6437662138b8d9f6bde2b3251b105abda57b8ba7e85959594c7  coe_node_esp32.bin
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
