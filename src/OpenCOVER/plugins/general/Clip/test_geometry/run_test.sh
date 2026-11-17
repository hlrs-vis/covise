#!/bin/bash
# Test the Clip plugin (v2) with a generated terrain + two "building" boxes.
#
#   ./run_test.sh                # target pre-applied from the config; add boxes via the menu
#   ./run_test.sh interactive    # start empty; pick target + boxes with the Clip menu
#
# Extra arguments are passed straight to cover, e.g.:
#   ./run_test.sh -c /path/to/your/cover-config.xml
#
# Requirements:
#   * the Clip plugin is built (ninja -C build Clip, and
#     cmake --build build --config RelWithDebInfo --target Clip for the
#     linux64opt cover binary). The plugin is loaded via the
#     "opencover://plugin/Clip" URL appended below.
#   * the OSG reader libraries on LD_LIBRARY_PATH, e.g.:
#         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/local_libs/lib
#     (needed to load the .obj test geometry and the plugin's dependencies).
#
# How the names are found: the objects are loaded from this directory, so their
# top-level scene-graph names are terrain.obj / building1.obj / building2.obj.
# The matching Clip setup is staged into ./.covconfig/plugin/Clip.toml, which
# the config manager searches before the installed config.

cd "$(dirname "$0")" || exit 1

MODE="setup"
if [ "${1:-}" = "interactive" ]; then
    MODE="interactive"
    shift
fi

# make sure the OSG reader / plugin dependency libs are reachable
if [ -d /home/hpcmdjur/local_libs/lib ]; then
    case ":$LD_LIBRARY_PATH:" in
        *":/home/hpcmdjur/local_libs/lib:"*) ;;
        *) LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/home/hpcmdjur/local_libs/lib" ;;
    esac
    export LD_LIBRARY_PATH
fi

STAGE_DIR=".covconfig/plugin"
mkdir -p "$STAGE_DIR"

if [ "$MODE" = "interactive" ]; then
    cp Clip_interactive.toml "$STAGE_DIR/Clip.toml"
else
    cp Clip_setup.toml "$STAGE_DIR/Clip.toml"
fi

# point the config manager at THIS directory's .covconfig (first in its search
# path), so the staged setup is read deterministically instead of the empty
# installed default
export COVCONFIG="$(pwd)/.covconfig"

# locate the cover launcher
if [ -n "$COVISEDIR" ] && [ -x "$COVISEDIR/bin/cover" ]; then
    COVER="$COVISEDIR/bin/cover"
elif command -v cover >/dev/null 2>&1; then
    COVER="$(command -v cover)"
else
    echo "could not find 'cover' (set COVISEDIR or put it on PATH)" >&2
    exit 1
fi

echo "Clip test (v2): mode=$MODE"
echo "  target: terrain.obj (pre-applied from config in setup mode)"
echo "Launching: $COVER terrain.obj building1.obj building2.obj opencover://plugin/Clip $*"
echo
echo "Tip: enable the 'Selection' nav mode, then use the 'Clip objects' menu:"
echo "  - 'Box from picked object' -> pick building1.obj / building2.obj,"
echo "    or 'Add box at hit point' with the interactor on the terrain,"
echo "  - in interactive mode: first 'Pick target (e.g. terrain)' -> terrain.obj."
echo
echo "Expected: the terrain shows a hole under each box; the orange ghost cubes"
echo "are selectable/movable (select one -> it turns red, move it -> the hole"
echo "follows, 'Clear' restores everything)."
echo

exec "$COVER" terrain.obj building1.obj building2.obj opencover://plugin/Clip "$@"
