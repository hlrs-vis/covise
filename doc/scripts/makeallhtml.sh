#! /bin/bash

find . -name 'doc' -and -type d -exec ${COVISEDIR}/doc/scripts/makemoddoc.sh \{\} \; 
