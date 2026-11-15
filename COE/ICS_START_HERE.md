# ICS team: update the node and record training data

Follow the parts in order. Each step says what you should see. **If what you see is
different, stop and use "If something goes wrong" at the end.** No programming knowledge
is needed. Don't edit any code, and don't move any jumper or wire except the one ESP
wire named in Part 2.

Time: about 45 minutes the first time (mostly installing one program).

What's new: the machine's speed is now chosen on the laptop, **1750 RPM or 3600 RPM**.
The two boards need new firmware for this (Parts 1 and 2). This is done once.

---

## Part 0. Prepare the laptop (once)

1. **Get the files.** Open https://github.com/amussab/Digital-Twin-Senior-Proj, click the
   green **Code** button, then **Download ZIP**, and unzip it, for example to your Desktop.
   (Or, if you use git: `git clone https://github.com/amussab/Digital-Twin-Senior-Proj` or `git pull`.)
   Everything you need is in the folder `COE`:
   - `COE/firmware/prebuilt/coe_node_stm32_LIVE.hex` (STM32 firmware)
   - `COE/firmware/prebuilt/coe_node_esp32.bin` (ESP32 firmware)
   - `COE/training/training_recorder.py` (the recording page)
2. **Install STM32CubeProgrammer** (free, from ST):
   https://www.st.com/en/development-tools/stm32cubeprog.html → "Get Software" → Windows version.
   ST asks for an email address (or a free account) before the download starts. Install it with
   the default options, and say **yes** if it asks to install USB drivers.
3. **Python 3.8 or newer** must be installed (`python --version` in a terminal shows it).
4. Use **Chrome or Edge** for Part 2.
5. Two USB cables that carry **data** (a charge-only cable won't work):
   one for the STM32 board and one USB-C for the ESP32 board.

**Before touching the hardware, take a photo of the node's wiring** (both boards, all wires).
If anything gets moved by accident, the photo shows how it was.

---

## Part 1. Update the STM32 (the big board, NUCLEO-H755ZI-Q)

1. Switch the **24 V supply ON**. The STM32 board gets its power from it. Don't move any jumper.
2. Connect the laptop to the STM32 board's **ST-LINK USB port**. It's the USB port at the end of
   the board with the ST-LINK part (the port used for programming), not the "USB USER" port at
   the other end.
3. Open **STM32CubeProgrammer**.
   - On the right, choose **ST-LINK**. Under it set **Mode: Under reset** and **Reset mode: Hardware reset**.
   - Click the green **Connect** button. You should see "Target connected" or the board's details.
4. In the left bar, click **Erasing & Programming** (the second icon, a download arrow).
   - **File path**: click **Browse** and choose `COE/firmware/prebuilt/coe_node_stm32_LIVE.hex`.
   - Tick **Verify programming** and **Run after programming**.
   - Click **Start Programming**. After about 20–60 s you should see **"File download complete"**
     and **"Download verified successfully"**.
5. Click **Disconnect** (top right) and unplug the STM32's USB cable.
6. Switch the **24 V supply OFF** (for Part 2).

---

## Part 2. Update the ESP32 (the small board, nanoESP32-C6)

**Rule: the ESP32 must never get power from USB and from the node at the same time.**
That's why the 24 V is off and its 5 V wire is unplugged during this part.

1. Check the **24 V supply is OFF**.
2. On the ESP32 board, find the wire on the pin marked **5V**. Pull out **only that one wire**
   (leave GND and the other wires where they are). Check your photo first.
3. Connect the laptop to the ESP32's USB-C port marked **CH343**.
4. In Chrome or Edge, open **https://espressif.github.io/esptool-js/**
   - **Baudrate**: 460800. Click **Connect**, choose the port that appears (it may be named
     "USB Single Serial", "CH343" or "USB-SERIAL"), then **Connect**. The page shows the chip:
     **ESP32-C6**.
   - Under **Program**: Flash Address **0x0**, and choose the file `COE/firmware/prebuilt/coe_node_esp32.bin`.
     (Delete any other file rows.)
   - Click **Program**. Wait until the log shows 100 % and "Leaving...". Click **Disconnect**.
5. Unplug the ESP32's USB cable.
6. Put the **5V wire back** on the ESP32's 5V pin, **pushed in firmly** (a loose 5V wire was a
   problem before: the ESP then stays dark).

---

## Part 3. Start the node and check it

1. Check both USB cables are unplugged. Switch the **24 V supply ON**.
2. Watch the ESP32's LED (within about 10 s):
   - red, briefly: starting
   - **blue**: its Wi-Fi **COE-NODE** is up
   - **green**: data is coming from the STM32. 
3. If it stays **blue** for more than 30 s, press the STM32's black **RESET** button once.

---

## Part 4. Record training data

1. Connect the laptop to the Wi-Fi **COE-NODE** (no password). It has no internet; that's normal.
2. Open a terminal in the folder `COE/training` and run:
   ```
   python training_recorder.py
   ```
   The first time, Windows asks about the firewall: click **Allow**.
3. Open **http://localhost:8000** in the browser. The pill at the top turns green:
   **"Data arriving: 1750 RPM, 4 of 4 channels OK"**.
4. **1. Machine speed**: click **1750 RPM** or **3600 RPM**, and set the **machine's speed
   controller to the same speed**. Wait for the green line **"Node confirmed: computing features
   for ... RPM"** (a few seconds).
5. **2. Choose the bearing condition**: Healthy, Inner race fault, Outer race fault, Ball fault or Cage fault.
   It must match the bearing that is installed in the machine.
6. **3. Record**: add a note if useful (load, which bearing), press **Start recording**, run the
   machine, press **Stop recording** when done. Start is refused while the node is still switching
   speed; just wait for "Node confirmed".
7. Repeat for every condition, **at both speeds** if your model should work at both.
8. **4. Recorded runs**: **Download all runs (CSV)**. The files are also saved in
   `COE/training/training_data/`. Copy that folder somewhere safe: it's not uploaded to GitHub.

About the data: one row per window, 32 features (`B1X_rms` … `B2Y_bpfi`, units in `COE/README.md`).
The `rpm` column is the speed the node used for that window. All four channels are independent
sensors (since 15 Nov 2026). Since 16 Nov 2026 the machine has new bearings (S6700) and the node
uses **bearing profile 3**: `bsf` is the ball band at 2 × BSF. Don't mix older recordings with these.

When finished: **Ctrl+C** in the terminal, then switch the 24 V supply off.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| STM32CubeProgrammer: "No STM32 target found" / can't connect | Is the 24 V on (Part 1 needs it on)? Try another USB cable (data, not charge-only). Hold the board's black RESET button, click Connect, release RESET after 1 s. |
| Programming failed or "verify" error | Click Connect again and repeat Part 1 step 4. |
| Web flasher shows no port | Try the ESP32's other USB-C port, or another USB-C cable. Close any other program using the port. |
| Web flasher: "Failed to connect" | Hold the ESP32's **BOOT** button, click Connect, release BOOT. Then Program. |
| ESP LED off after Part 3 | 24 V off, push the ESP's 5V wire in firmly, 24 V on. |
| ESP LED keeps flashing red | The 5 V supply dips. Stop and contact Abdullah. |
| Page says "No data yet" but COE-NODE is connected | Run this once in an **administrator** terminal, then restart the script: `netsh advfirewall firewall add rule name="COE node TCP 8000" dir=in action=allow protocol=TCP localport=8000 profile=any` |
| Speed stays "Switching the node to 3600 RPM" for more than 30 s | The return wire ESP GPIO19 → STM32 PD6 may be missing. Record at **1750** only and tell Abdullah. |
| "Cannot use port 8000" | The script is already running in another terminal. Close it first. |
| Anything else | Stop. Take photos of the boards (LED colours) and a screenshot, and contact Abdullah. Don't change wiring or jumpers. |

Details about the node (for COE): `COE/README.md`, `COE/firmware/README.md`.
