# Surfboard navigation provider

The plugin reads one text record per line from a serial port, for example
`Angle: -154.60 deg | GPIO4/5: 3 (0b11)`. It parses the signed degree value
after `Angle:` and the decimal GPIO state after `GPIO4/5:`. A zero GPIO state
means released; any nonzero state means pressed. CR/LF terminates a record.

The plugin uses COVISE's cross-platform `covise::SerialCom`; configure the
device as `COM7`/`\\\\.\\COM7` on Windows or `/dev/ttyUSB0` on Linux.
Only the OpenCOVER master opens the serial device; its navigation transform is
synchronized to the slaves.

Add the plugin to the OpenCOVER plugin list and configure it in the COVISE
configuration file, inside `<COVER><Plugin>`:

```xml
<Surfboard device="/dev/ttyUSB0" baudRate="115200"
           maxSpeed="1.5" acceleration="1.0" deceleration="1.5"
           steeringRate="0.7" turnRate="4.0" deadZone="2.0"
           turnThreshold="45.0" buttonRequired="true" />
```

On Windows, change `device` to the encoder's COM port. Speeds are in m/s,
accelerations in m/s², angular rates in radians/s, and angles in degrees.
With the navigation provider enabled, the button must be pressed to move by
default. Small signed angles steer and move laterally (`-x` for negative and
`+x` for positive); they do not move in Y. Angles beyond `turnThreshold`
rapidly turn the board toward -180° or +180° and then move forward along +Y.
Each relative movement is applied to the scene transform currently returned by
`VRSceneGraph`, following the Skateboard plugin's transform update pattern.

The sensor angle at the moment navigation is enabled becomes the virtual zero
heading. Smaller angles directly steer the heading left or right. Crossing the
45° threshold starts a complete turn to the matching ±180° heading. The
Surfboard menu provides live acceleration and deceleration dials for tuning
that turn profile.
