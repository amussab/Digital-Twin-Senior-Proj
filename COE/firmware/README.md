# COE node firmware

The code that runs on the node's two boards. The ICS team doesn't need to touch it; the scripts in
`../training` and `../inference` are all they use. This folder is for COE, and for anyone who
needs to rebuild or reflash a board.

| Folder | Board | What it does |
|---|---|---|
| [`stm32/`](stm32/README.md) | NUCLEO-H755ZI-Q (STM32H755) | Reads the AD7606 ADC (4 channels, 30 kS/s), filters, computes the 32 features per window, sends them to the ESP32 over UART |
| [`esp32/`](esp32/README.md) | nanoESP32-C6 | Runs the Wi-Fi **COE-NODE**, finds the receiver laptop, forwards each window as HTTP JSON |

The two folders must stay **side by side**: the STM32 build uses two files from `esp32/main`
(the shared packet format).

## Build and flash

- **STM32** (STM32CubeCLT, the board on USB): in `stm32/`,
  `powershell -ExecutionPolicy Bypass -File .\tools\flash.ps1 -Mode LIVE`
  (`tools\build.ps1 -Mode LIVE` builds without flashing). LIVE is the mode the node runs.
- **ESP32** (ESP-IDF v6.1, the board on USB): in `esp32/`, `idf.py build flash`.
  As cloned, it runs only its own Wi-Fi COE-NODE. To also join a phone hotspot, copy
  `main/config/connection_settings.example.h` to `connection_settings.local.h` and fill it in.
  Git ignores that file, so passwords stay private.
- After reflashing the ESP32, press the STM32's RESET button.

Details, wiring, tests and measured results: `stm32/README.md` and `esp32/README.md`.
In those READMEs, `coe_firmware` means `stm32/` and `coe_esp32` means `esp32/` (their original folder names).

## Settings you might change

- Machine speed: `stm32/CM7/App/config/live_config.h`, `COE_LIVE_NOMINAL_RPM` (60–3600, now 250).
  Rebuild and flash the STM32.
- Auto-start on power-up: same file, `COE_LIVE_AUTOSTART` (1 = on).
