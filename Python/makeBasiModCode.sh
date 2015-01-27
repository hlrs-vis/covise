#! /bin/bash
source "${COVISEDIR}"/.covise.sh
source "${COVISEDIR}"/scripts/covise-env.sh

"${PYTHON_HOME}"/bin/python3 "${COVISEDIR}"/Python/makeBasiModCode.py "$@"
