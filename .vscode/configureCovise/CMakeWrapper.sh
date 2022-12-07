#! /bin/bash
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# while read p; do
#   export "$p"
# done <$SCRIPT_DIR/covise.env
# echo this is working!!!!!!!!!!!!!
# echo "$@"
# @echo off
# set last=0
# setlocal ENABLEDELAYEDEXPANSION
# for %%x in (%*) do (
#    if "%%x" EQU "-DCMAKE_BUILD_TYPE:STRING" (
#     set last=1
#    )
#    IF !last! == 1 (
#       if "%%x" EQU "Debug" (
#         set ARCHSUFFIX=$BASEARCHSUFFIX
#       ) else if "%%x" EQU "Release" (
#         set ARCHSUFFIX=zebuopt
#       ) else if "%%x" EQU "RelWithDebInfo" (
#         set ARCHSUFFIX=zebuopt
#       ) else if "%%x" EQU "MinSizeRel" (
#         set ARCHSUFFIX=zebuopt
#       )
#    )
# )

exec cmake "$@"


 