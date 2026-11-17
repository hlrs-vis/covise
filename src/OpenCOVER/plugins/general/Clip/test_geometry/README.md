# Clip plugin — test scenes

Generated geometry to test the Clip plugin without a COVISE data source.

## Scenes

### Classic: wavy terrain + boxes

| file | role | geometry |
|---|---|---|
| `terrain.obj` | target | 40x40 wavy heightfield (central mound + ripples), z in `[0, ~2]` |
| `building1.obj` | void | 3x3 box at the mound peak, `z -1..7` (straddles the surface) |
| `building2.obj` | void | 2.5x2.5 box on a slope, `z -1..4.5` (straddles the surface) |

The boxes deliberately extend both below and above the terrain so the terrain
is actually cut where they overlap.

### DSM: photogrammetry-style multi-geode target

`dsm_terrain.osgb` is ONE object with **9 geodes** (a 3x3 tile grid), like a
real photogrammetry model. The surface is a dense heightfield with building
footprints modelled as flat-topped spikes that are **part of the surface**
(a digital surface model) — the boxes cover those spikes:

| file | role | geometry |
|---|---|---|
| `dsm_terrain.osgb` | target | group `dsm_terrain` with geodes `tile0..tile8`; spikes at (0,0) h=5.5, (5,-4) h=3.8, (-6,5) h=4.6 |
| `building1.obj` | void | 3x3 box at (0,0), covers the first spike |
| `building2.obj` | void | 2.5x2.5 box at (5,-4), covers the second spike |

The third spike has no void box on purpose: it must remain uncut.

## How to run

```
./run_test.sh                # classic scene, setup auto-applied
./run_test.sh interactive    # start empty, pick with the Clip menu
./run_dsm_test.sh            # DSM scene, setup auto-applied
./run_dsm_test.sh interactive
```

Extra args go to `cover`, e.g. pick a config file:

```
./run_test.sh -c /path/to/your/cover-config.xml
```

The scripts `cd` into this directory (so the objects keep their file names as
scene names), stage the matching `Clip.toml` into `./.covconfig/plugin/`, and
set `COVCONFIG=$(pwd)/.covconfig` so the staged setup is read deterministically
instead of the (empty) installed default.

### Interactive mode

1. Enable the **Selection** navigation mode (laser point + action button).
2. In the **Clip objects** menu press **Pick target (e.g. terrain)** and pick
   the terrain.
3. Press **Add void object** and pick `building1.obj`, then `building2.obj`.
4. A hole appears where each box overlaps the terrain; the boxes stay hidden
   but pickable.

## What to check

- The terrain renders **normally (not black)** when it is the target.
- Each building cuts a hole where it actually pokes through the terrain; you
  can see through the hole into the scene.
- **Only the part of the terrain inside the box is removed**; terrain behind
  the box (further down the view ray) stays visible.
- The buildings are **not** rendered in front of the terrain — they are still
  depth-tested, so the part of a building behind the terrain is hidden.
- **Clear all** removes the holes; **Remove void object** removes one.
- DSM scene: the cut spans **tile borders** without gaps (all 9 geodes are
  clipped), the unboxed spike stays, and the terrain behind the boxes is kept.

## Regenerating

```
python3 generate_test_geometry.py   # rewrite all .obj files (incl. DSM tiles)
./build_dsm_osgb.sh                 # re-assemble dsm_terrain.osgb from the tiles
```

`run_dsm_test.sh` does both automatically if `dsm_terrain.osgb` is missing.

## Multi-geode test tool

The DSM `.osgb` is built by `mk_osgb.cpp` (a tiny OSG program the build script
compiles to `/tmp/mk_osgb` if the binary is missing): it loads the tile
`.obj` files and writes one top-level `Group` ("dsm_terrain") with one geode
per tile — the same structure a photogrammetry `.osgb` has. Use it to convert
any set of `.obj` tiles into a single multi-geode `.osgb`:

```
LD_LIBRARY_PATH=... /tmp/mk_osgb out.osgb myModel tile0.obj tile1.obj ...
```
