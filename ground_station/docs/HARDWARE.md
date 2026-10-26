# Hardware and information flow

## Existing context

The [Basement Creations video adaptation notes](VIDEO_REFERENCE.md) and local creator file bundle supplement this guide. Its F405 Mini wiring and AirPort settings differ from our F405 AIO/native MAVLink path; consult the adaptation notes before using its setup PDF.

Earlier project discussion identifies Betaflight 4.5.1 / `SPEEDYBEEF405AIO`, RadioMaster Pocket and an FPV GPS around ₹1,900. The exact receiver, ELRS version, GPS model/compass, analogue video receiver and capture card remain unconfirmed. Nothing in this build changes your firmware or flight-controller parameters.

## Recommended firmware

Use ArduCopter for ground-issued navigation requests. Its GUIDED mode supports velocity and position setpoints while the FC closes the stabilization loop. Your board has an official ArduPilot target, but firmware feature availability and exact revision still need checking. Back up Betaflight configuration, ESC settings, receiver firmware/settings and motor mapping before migration. The earlier backup had an invalid `osd_units` value and custom motor order; those need resolving before relying on a restoration file.

ArduPilot documents initial `with_bl.hex` flashing via DFU and subsequent `.apj` updates for this board. Reflashing requires fresh configuration and calibration; Betaflight settings do not translate directly.

## Video route

```text
FPV camera → FC analogue OSD → video transmitter (VTX)
    → matching analogue video receiver / goggles AV OUT
    → correct composite AV cable → USB capture card → laptop
```

The USB capture card digitizes the receiver's video. It is not a 5.8 GHz RF receiver. An HDMI-only dongle cannot directly accept yellow-RCA composite video. If your goggles only output HDMI, choose the HDMI route instead; identify the connector before buying anything. TRRS AV cable pinouts vary by receiver/goggle manufacturer.

## Low-cost capture candidate

On 9 October 2026, [Probots EasyCAP A072](https://probots.co.in/easycap-a072-usb-2-0-audio-video-capture-adapter.html) is listed at **₹349 including GST**, excluding shipping. The seller lists composite RCA/S-Video, PAL 720×576 at 25 fps, NTSC 720×480 at 30 fps, and Windows 10/11 support. Treat these as seller claims. Chipsets and drivers vary among inexpensive EasyCAP products. Confirm your Windows version, exact unit's driver support and return terms, and measure latency; no measured latency is advertised in the cited specifications. This is a low-cost candidate for bench testing, not a verified flight-control capture device.

Start capture settings at the camera's native standard (PAL: 720×576 / 25; NTSC: 720×480 / 30). High requested resolutions cannot add detail to analogue video. Configure capture format in the vendor tool if necessary, close that tool, then use this project.

## Telemetry and laptop-to-drone commands

```text
GPS / FC sensors → ArduCopter → full TX/RX UART → ELRS receiver
    ↔ ELRS RF link ↔ RadioMaster transmitter module
    ↔ TX Backpack Wi-Fi / supported external-module USB ↔ laptop
```

ExpressLRS supports bidirectional MAVLink. Conventional CRSF telemetry can show sensors on EdgeTX; that does not automatically expose a laptop MAVLink endpoint. Internal RadioMaster modules require compatible Backpack Wi-Fi forwarding. The official minimums for this path are ELRS TX/RX 3.5.0 and Backpack 1.5.0; check the current version-specific procedure. ESP-based hardware is needed. Some recent versions automate RX protocol selection differently.

In the ELRS Lua script, use Backpack `Telemetry = WiFi`; manually enabling firmware-update Wi-Fi is a different operation. Join the Backpack network or its shared home network. The documented GCS receive port is UDP 14550. Do not let Mission Planner and this project compete for the same UDP/serial endpoint; use one receiver or an explicitly configured MAVLink router.

For ArduPilot, the official ELRS guide specifies the chosen receiver port `SERIALx_PROTOCOL=2`, `SERIALx_BAUD=460`, `RSSI_TYPE=5`, and reduced stream rates. Use the exact documented board port, not a guessed UART. The F405 AIO documentation maps UART6 to SERIAL6 (DMA-enabled), UART5 to GPS, USB to SERIAL0, and notes UART2 has only an inverted RX pad. Full MAVLink needs two directions: receiver TX → FC RX; receiver RX → FC TX; common ground and correct supply voltage. Do not blindly apply these values before wiring and hardware confirmation.

GPS connects to the FC, not directly to the capture card. MAVLink then carries position/fix/satellite information to the laptop. A cheap FPV GPS may have no compass; the F405 AIO has no internal compass. Confirm the module's markings, pins, protocol and voltage, and the heading sensor/estimator plan. GPS position alone is not a trustworthy heading while hovering.

## What video cannot provide alone

OSD OCR can recover visible text with errors, but GPS coordinates are unavailable if not rendered. A person bounding box is a pixel location, not a geographic position. Accurate victim coordinates need camera calibration, time-synchronized aircraft pose, height/terrain and ray projection or another ranging method. Monocular FPV video alone does not establish obstacle distance or safe free space. Analogue video contains no authenticated flight telemetry channel.

References: [ELRS MAVLink](https://www.expresslrs.org/software/mavlink/), [board support](https://ardupilot.org/copter/docs/common-speedybeef405aio.html), [Guided commands](https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html).
