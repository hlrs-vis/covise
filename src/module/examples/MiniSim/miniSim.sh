#!/bin/sh
cd $COVISEDIR/src/application/examples/MiniSim

export CO_SIMLIB_CONN=$1
echo "CO_SIMLIB_CONN: $CO_SIMLIB_CONN"

./miniSim
