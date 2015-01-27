@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM renames a file or directory by first resetting
REM the hidden and system attributes
REM necessary e. g. for renaming subversioned directories
REM *******************************************

IF "x%1x" EQU "xx" GOTO USAGE
IF "x%2x" EQU "xx" GOTO USAGE
IF "x%3x" EQU "xx" GOTO USAGE

PUSHD
CD /D %1
ATTRIB -S -H %2
RENAME %2 %3
POPD
EXIT /B 0

:USAGE
ECHO ...
ECHO renames a file or directory by first resetting
ECHO the hidden and system attributes
ECHO necessary e. g. for renaming subversioned directories
ECHO ...
ECHO usage:
ECHO %0 
ECHO    [path in which to execute renaming]
ECHO    [current file or folder name]
ECHO    [target name]
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
EXIT /B 1