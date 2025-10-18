# ColorAnim Plugin

## Description
This OpenCover plugin animates vertex colors across 275 brain models. It loads the models, extracts their vertex colors, and provides smooth animation with configurable playback speed.

## Brain Model Requirements
- 275 3D models (PLY format, binary little endian)
- Each model: 7501 vertices, 14998 faces
- Models differ only in vertex colors (RGB as uchar)
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
- `cortex_001.ply`, `cortex_002.ply`, ..., `cortex_275.ply` (all with 3-digit padding)

## UI Controls

The plugin adds a "Brain Color Animation" menu with:

- **Play/Pause** button - Start/stop the animation
- **Reset** button - Jump back to frame 0
- **Speed** slider - Control animation speed (0.001x to 0.1x, very slow to slow)
- **Ping-Pong Mode** button - Toggle between loop mode and back-and-forth mode
- **Flip Normals (Inside View)** button - Flip normals to view brain from inside
- **Interpolation** group - Select color interpolation mode:
  - **Linear** - Standard linear interpolation (default)
  - **Smoothstep** - Smooth acceleration/deceleration (3t² - 2t³)
  - **Smootherstep** - Even smoother curve (6t⁵ - 15t⁴ + 10t³)
  - **Ease In-Out** - Cosine-based smooth easing
  - **Cubic** - Cubic ease in-out with sharp acceleration
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
