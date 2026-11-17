#!/bin/sh
# Assembles the generated dsm_tile_*.obj (photogrammetry-style DSM heightfield
# with building spikes) into one multi-geode object, dsm_terrain.osgb:
#
#   dsm_terrain (Group)
#     tile0 (Geode)
#     tile1 (Geode)
#     ...  (9 tiles)
#
# This mimics a real photogrammetry .osgb: one loaded object made of many
# geodes. Run generate_test_geometry.py first to (re)generate the tiles.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
TOOL=/tmp/mk_osgb

if [ ! -x "$TOOL" ]; then
    g++ -std=c++17 -o "$TOOL" "$HERE/mk_osgb.cpp" \
        -I/home/hpcmdjur/local_libs/include \
        -L/home/hpcmdjur/local_libs/lib \
        -losgDB -losg -losgUtil -losgViewer -losgGA -lOpenThreads
fi

cd "$HERE"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/hpcmdjur/local_libs/lib" "$TOOL" \
    dsm_terrain.osgb dsm_terrain \
    dsm_tile_0_0.obj dsm_tile_0_1.obj dsm_tile_0_2.obj \
    dsm_tile_1_0.obj dsm_tile_1_1.obj dsm_tile_1_2.obj \
    dsm_tile_2_0.obj dsm_tile_2_1.obj dsm_tile_2_2.obj
