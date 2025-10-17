# ColorAnim Plugin

## Description
This OpenCover plugin animates vertex colors across 275 brain models. It loads the models, extracts their vertex colors, and provides smooth animation with configurable playback speed.

## Brain Model Requirements
- 275 3D models (OBJ format)
- Each model: 7501 vertices, 14998 faces
- Models differ only in vertex colors
- Same geometry structure across all models

## Configuration

Add to your COVISE config file:

```xml
<COVER>
  <Plugin>
    <ColorAnim>
      <ModelPath value="/path/to/brain" />
    </ColorAnim>
  </Plugin>
</COVER>
```

The plugin expects files named:
- `brain_001.obj`, `brain_002.obj`, ..., `brain_275.obj`
- OR `brain_0.obj`, `brain_1.obj`, ..., `brain_274.obj`

## UI Controls

The plugin adds a "Brain Color Animation" menu with:

- **Play/Pause** button - Start/stop the animation
- **Reset** button - Jump back to frame 0
- **Speed** slider - Control animation speed (0.1x to 10x)
- **Frame** label - Shows current frame number

## Usage

1. Build COVISE with the plugin
2. Configure the model path in your config
3. Start OpenCover
4. Open the "Brain Color Animation" menu
5. Click "Play" to start the animation

## Implementation Details

- Loads all 275 color arrays into memory
- Interpolates between frames for smooth animation
- Base animation speed: 30 fps
- Color interpolation is linear (LERP)
- Automatically loops when reaching the last frame
