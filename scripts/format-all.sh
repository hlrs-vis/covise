#! /bin/bash

find "${COVISEDIR}/src" \
   -name '*.h' -o -name '*.hpp' -o -name '*.c' -o -name '*.cpp' -o -name '*.cxx' \
   -o -name '*.H' -o -name '*.C' -o -name '*.cc' -o -name '*.CC' \
   | xargs "${COVISEDIR}/scripts/prepend-license.sh"
exit 0

find "${COVISEDIR}/src" \
   -name '*.h' -o -name '*.hpp' -o -name '*.c' -o -name '*.cpp' -o -name '*.cxx' \
   -o -name '*.H' -o -name '*.C' -o -name '*.cc' -o -name '*.CC' \
   | xargs "${COVISEDIR}/scripts/indent.sh"
exit 0
