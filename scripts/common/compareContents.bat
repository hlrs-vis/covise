@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author Harry Trautmann
REM ******************************************
REM compares the contents of 2 files for
REM if each line of the left file is contained 
REM in any line of the right file
REM lines not found are output
REM ******************************************

SETLOCAL ENABLEDELAYEDEXPANSION

IF /I "x%1x" EQU "xx" GOTO USAGE
IF /I "x%2x" EQU "xx" GOTO USAGE

SET _LEFT=%1
SET _RIGHT=%2
SET _UNIXUTILS=%3

SET _TMPFILE1="%TMP%\~compareContentstmp1.txt"

FOR /F "delims=^" %%G IN (%_LEFT%) DO (
   %_UNIXUTILS%\grep.exe -c -e "%%G" %_RIGHT% > %_TMPFILE1%
   SET /P _NUM=<%_TMPFILE1%
   IF "x!_NUM!x" EQU "x0x" ECHO %%G
)
REM ECHO left file: %_LEFT%
REM ECHO right file: %_RIGHT%

GOTO END

:USAGE
ECHO ...
ECHO usage:
ECHO %0
ECHO    [left file: path + filename]
ECHO    [right file: path + filename]
ECHO    [path to UnixUtils] -- optional, if not given, unix utils 
ECHO       are assumed to be in path
ECHO ...
ECHO compares the contents of 2 files for if each line of the left file 
ECHO is contained in any line of the right file
ECHO lines not found are output
ECHO ...
ECHO called executables:
ECHO     %_UNIXUTILS%\grep.exe
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO author Harry Trautmann
ECHO ...
ENDLOCAL
EXIT /B 1

:END
DEL %_TMPFILE1%
ENDLOCAL
EXIT /B 0