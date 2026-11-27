# ESP32 sync output

PlatformIO Arduino project for an ESP32 DevKit (`esp32dev`). GPIO 32 and GPIO 33
produce the same signal, at a configurable frequency (default 60 Hz):

- `normal`: 50% duty square wave.
- `pulse`: one high pulse per period, followed by a low interval.
- `vga`: positive vertical sync pulse each period.
- `vga-`: negative vertical sync pulse each period.

Inversion is a separate setting and applies to every waveform. Use `INVERT on`
or `INVERT off`; `MODE invert` remains a shortcut for enabling inversion.

Connect the serial monitor at 115200 baud and send newline-terminated commands:

```text
MODE pulse
WIDTH 500
FREQ 60
MODE vga
INVERT on
STATUS
MODE normal
INVERT off
```

`WIDTH` is in microseconds and accepts 1 through one less than the current
period. `FREQ` accepts integer values from 1 through 1000 Hz. If a frequency
change makes the saved pulse width too long, the width is reduced to one
microsecond less than the new period. The waveform mode, inversion setting,
pulse width, and frequency are stored in ESP32 NVS (non-volatile storage) and
restored at boot. Defaults are `normal` mode, inversion off, 60 Hz, and a
1000 us pulse width.

Build and upload with PlatformIO:

```sh
pio run
pio run --target upload
pio device monitor
```
