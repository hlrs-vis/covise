#! /bin/bash
source "${COVISEDIR}"/.covise.sh
source "${COVISEDIR}"/scripts/covise-env.sh

make -C bin/vr-prepare "$@"
