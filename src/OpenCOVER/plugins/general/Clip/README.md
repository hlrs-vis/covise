# Clip plugin for OpenCOVER

Cuts holes in one object (the *target*, e.g. a terrain) where other selected
objects (the *voids*, e.g. buildings) overlap it. By default the void objects
are **hidden** from the rendered scene but stay **pickable and editable**:
you can still select them, move them with the Move plugin, and the hole
follows the moved object.

## How it works

The plugin uses the GL **stencil buffer** instead of a custom fragment shader,
so the target keeps its own rendering (e.g. the heightmap terrain shader).
Every frame the scene is drawn in three ordered stages (render bins):

1. **Target depth pre-pass** (bin `-2`): a "shadow" clone of the target is
   rendered into the depth buffer only (color write off, lighting off, depth
   write on). This gives the void pass something to depth-test against.
2. **Void stencil pass** (bin `-1`): a "shadow" clone of each void object is
   rendered into the stencil buffer only (color write off, **depth write
   off**). Each fragment depth-tests against the target shadow and writes
   stencil `1` only where it passes (i.e. where the void is actually in front
   of / inside the target).
3. **Normal scene** (bin `0`): the whole scene renders unchanged, except the
   target carries two extra attributes: a stencil test `EQUAL 0` (write mask
   0) so it is skipped where the voids wrote (the hole), and a `LEQUAL` depth
   test so it still passes against the depth its own shadow wrote (otherwise
   it would fail its own depth and disappear).

### Shadow clones

The shadows are deep copies of the original geodes. Each frame the plugin
re-resolves the current geode by object name and keeps every shadow as a
sibling **under the geode's current parent**:

- **Move plugin**: wrapping an object in its (helper) `MatrixTransform`
  re-parents the geode, so the shadow is moved with it — the clone carries the
  same local matrix and stays exactly on the object.
- **COVISE object updates** (replace): the new geode is re-cloned and the
  shadow is re-created.

The shadows' node masks remove them from picking, intersection, collision and
navigation, so they are invisible and do not disturb the scene.

### Hidden but editable voids

Hiding is done purely with **render state** on the void geode, never by
clearing the `Isect::Visible` bit (COVER's picking rejects a hit if the node
or one of its ancestors is not `Visible`, so that would make the object
ungrippable):

- color write off, depth write off, stencil ignored (OVERRIDE attributes)
- the `Collision`/`Walk`/`Touch` mask bits are removed, so lasers, walking and
  collision pass through the invisible object
- `Pick` (and `Visible`) stay set, so Selection and Move can still grab it

A void that is **currently selected or being moved** (a *Selection* helper
node is above its geode) is temporarily shown again, so you always see the
object you are editing; it hides again as soon as the selection is cleared.
The toggle is available as the **"Hide void objects"** button and as the
`Setup.hideVoids` config value.

The cut is **depth-accurate**: the target is removed only where a void object
is actually in front of (or inside) it from the current view. A void hidden
behind the target does **not** cut a hole, and a void does not render in front
of the target — it is still depth-tested against the target in the normal pass.

## Enabling the plugin

Add the plugin to the `<Plugin>` section of your cover config file, e.g.
`config/config.cover.xml`:

```xml
<Plugin>
    ...
    <Clip value="on" />
    ...
</Plugin>
```

Build it with the target name `Clip`:

```
cmake --build build --target Clip
```

This produces `libClip.so` in `<archdir>/lib/OpenCOVER/plugins/`
(`linux64/` or `linux64opt/` depending on the build type).

## Interactive use

1. Enable the **"Selection"** navigation mode (the standard laser-point
   picking; press the action button to pick).
2. Open the **"Clip objects"** menu (plugin menu).
3. Press **"Pick target (e.g. terrain)"**, then point at the object you want
   cut and press the action button.
4. Press **"Add void object"** and pick each object that should produce a
   hole. The mode stays active, so you can pick several voids in a row.
5. Use **"Remove void object"** (or the void list) to delete a void again, and
   **"Clear all"** to reset everything.
6. Toggle **"Hide void objects"** to show/hide the void objects (default:
   hidden). While a void is selected (e.g. with the Move plugin) it is shown
   anyway.

The hole appears as soon as both a target and at least one void are set.

## Config file setup (`config/plugin/Clip.toml`)

Instead of (or in addition to) picking interactively, the target and voids can
be declared in the plugin's toml config:

```toml
[Setup]
target = "terrain.obj"
voids = ["building1.obj", "building2.obj"]
hideVoids = true
```

Names are matched against the scene graph's top-level node names:

- **COVISE objects**: the object name (`"terrain"`).
- **File-loaded objects** (e.g. `cover terrain.obj ...`): the top-level node
  is named by its load URL (e.g. `terrain.obj` or `/abs/path/terrain.obj`), so
  use the file name or its path as loaded. A plain prefix also works.

The config is read on plugin start and re-applied automatically whenever the
file changes (or objects load in). A manual pick takes the setup over from the
config; the **"Apply setup from config"** button puts the config back in
charge. `Setup.hideVoids` (default `true`) controls whether the void objects
are hidden; the "Hide void objects" button writes it back, so it is shared.

## Notes and limitations

- The target and the voids must be separate objects (self-cutting is not
  defined).
- The cut is a visual stencil effect on the target object; the underlying
  geometry is not modified.
- Hidden voids are still **collision-free** (the collision/walk/touch bits are
  cleared), which is usually what you want for buildings behind the terrain —
  you can walk/laser through them.
- Transparent targets or voids are best-effort: the pre-passes render with
  depth only, so alpha-blended edges may not cut perfectly.
- Requires the stencil buffer that COVER requests by default
  (`COVER.StencilBits`, default 1 bit — enough for this binary test).

## Testing

See `test_geometry/` for a generated terrain + two "building" boxes and a
`run_test.sh` that loads them with a pre-applied (or interactive) Clip setup:

```
test_geometry/run_test.sh            # setup auto-applied, holes visible at once
test_geometry/run_test.sh interactive # start empty, pick with the Clip menu
```

Things to check:

- terrain shows a hole at each building, buildings are not visible
- select a building (Selection nav mode) → it becomes visible while selected
- move it with the Move plugin → the hole follows, it hides again afterwards
- "Hide void objects" button toggles the visibility of all voids
