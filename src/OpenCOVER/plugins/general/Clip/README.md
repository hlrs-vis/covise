# Clip plugin for OpenCOVER (v2)

Cuts axis-aligned, world-space rectangular "voids" (holes) out of one or more
target objects (e.g. photogrammetry terrain). v2 replaces the old v1
implementation (stencil buffer + shadow clones + three extra render bins)
with a **per-pixel GLSL 150 cut**:

* the vertex shader passes each vertex's **ObjectsRoot-space position**
  (via a per-geode `clipWorldMatrix` uniform) and its normal through to the
  fragment shader,
* the fragment shader tests that position against up to 16 axis-aligned
  boxes (fed as `clipBoxMin[i]` / `clipBoxMax[i]` uniforms) and **discards**
  the fragments that fall inside any box.

Discarding (rather than fading) keeps the target in the **opaque** render
bin, so depth writing and sorting stay as they were and whatever is behind
the hole shows through. No stencil bits, no shadow clones, no extra render
passes.

Because the fragment shader takes over the fixed-function pipeline for the
clipped geodes, it also reproduces the target's own look: material (or
vertex color) times an optional base texture, lit by the scene headlight
(ambient + Lambert). Everything the shader overrides (material, texture
binding, depth state) is saved per geode and **restored exactly** when the
target is removed or the plugin is destroyed.

## The boxes

The clip boxes are **first-class scene nodes** (`ClipBoxNode`, an
`osg::MatrixTransform` with translucent ghost-cube geometry) under the
ObjectsRoot — i.e. ordinary COVER objects. You can select them with the
Selection plugin and move them with the Move plugin; the cut follows
automatically, because each frame the plugin re-derives every box's world
AABB from its (possibly moved) world transform and pushes it into the
shared shader uniforms. The same holds for the targets: move the terrain
and the cut stays in place, in world space.

* **auto**: pick an object → a box matching its world AABB is created,
* **manual**: *toggle* "Add box at hit point" on, then **every click** on an
  object drops a default 10 × 2 × 10 box with its bottom corner at the hit
  point — repeat as often as you like, untoggle when done. Use it to place
  boxes at arbitrary spots, not just where an object already is.
* **config**: `boxes = ["minX,minY,minZ,maxX,maxY,maxZ", ...]` in the
  plugin's `[Setup]` config defines boxes directly — no object in the scene
  is needed (see below).

The box AABB is computed from the box's world transform every frame, so
boxes placed at an angle (by moving them) still cut as the *axis-aligned
bounding box* of their current extent.

### Resizing a box

Select a box with the Selection plugin (it highlights red). The **Box size**
sliders in the menu then show
the box's current world extent (X, Y, Z). Drag a slider to resize the box
around its center; the cut follows on the next frame. The sliders update
live while the box (or the target) is being moved. They work in the VR and
tablet UIs as well as on the desktop (unlike text fields, no keyboard is
needed). If a box is larger than a slider's range, the range expands
automatically.

## UI (menu: "Clip objects" → "Cut holes with boxes")

| Button | Action |
|--------|--------|
| **Pick target** (toggle) | enter/leave target-picking mode: select an object (e.g. the terrain) to add/remove it as a clip target. Multiple targets are supported. |
| **Box from picked object** (toggle) | enter/leave box-from-object mode: select an object to create a clip box matching its world AABB. |
| **Add box at hit point** (toggle) | while enabled, **each click** on an object drops a default 10 × 2 × 10 box with its bottom corner at the hit point. Untoggle when done — place as many boxes as you like, anywhere you can point. |
| **Remove last box** | removes the most recently created box. |
| **Box size X / Y / Z** | sliders showing the world extent of the currently *selected* box; dragging one resizes that box around its center (see "Resizing a box"). |
| **Show box ghosts** (toggle) | show/hide the translucent ghost cubes (configurable, persists per model). |
| **Clear** | removes all boxes and all targets (restoring their original rendering). |
| **Apply setup from config** | re-apply the targets listed in the plugin config. |

The label at the bottom shows the current target list and box count.
Selecting a box with the Selection plugin highlights it in red.

## Configuration

Per-model (`[Setup]` in the model's `*.toml`):

```toml
[Setup]
targets = ["terrain"]        # top-level object or sub-node names to clip
boxes = ["-5,-5,0,5,5,8"]    # clip boxes: "minX,minY,minZ,maxX,maxY,maxZ" (world coords)
showBoxes = true             # ghost cubes visible
```

* **`boxes`** defines the clip boxes **independently of any object in the
  scene** — pure min/max corners in world (ObjectsRoot) space. Each entry is
  one box: a **quoted** string `"minX,minY,minZ,maxX,maxY,maxZ"`. The max
  corner may also be given before the min corner; the plugin sorts it.
  This is the way to place a void where there is no object to model it
  with (e.g. a rectangular foundation pit in the terrain).
* When the config is (re)applied, configured boxes are created, moved or
  resized to match (matched against existing boxes by center proximity, so
  an edited box keeps its identity), and boxes that no longer appear in the
  config are removed. Boxes you created **manually** in the scene are kept
  — only boxes the config previously knew about are removed.

Targets are stored by **name**. File-loaded objects are named by their
URL, so the config name is matched by exact name, basename, or prefix
(e.g. `terrain` matches `/path/to/terrain.obj`). Manually made changes
(remove target, add box) stop the config from being auto-applied; press
"Apply setup from config" to hand control back to the config.

## Limitations

* Boxes are **axis-aligned** (in world/ObjectsRoot space). Rotating a box
  still cuts with its axis-aligned bounding box.
* At most **16 boxes** are active in the shader (fixed uniform array size);
  extra boxes are ignored (the status label still counts them).
* The fragment shader replaces fixed-function shading for the clipped
  geodes, so it approximates the target's look (diffuse + ambient, no
  specular, no texture environment). Targets with custom shaders (e.g. the
  VRML heightmap) lose those effects while clipped.
* The cut is a **soft boundary at pixel resolution**: box edges are as crisp
  as the screen sampling, not as a hard geometric intersection.
* Targets are clipped at the **node** level (whole top-level object or a
  named sub-node), not per-drawable.
