@ECHO OFF

REM ^(C^) Copyright 2009 VISENSO GmbH
REM author Harry Trautmann
REM *****************************************
REM sets Paths in *.copr files
REM *****************************************

SETLOCAL ENABLEDELAYEDEXPANSION

IF "x%1x" EQU "x--helpx" GOTO USAGE
IF "x%1x" EQU "x-hx" GOTO USAGE
IF "x%1x" EQU "x/?x" GOTO USAGE

SET _NEWPATH=%1
SET _FPAT=%2
SET _PPAT=%3

ECHO Setting all path information in files ...

ECHO ... DOES NOT WORK YET !!! ...
type sed.exe s/:S'(.\/|\/)[^']*:/%_NEWPATH%/g > %_TMPFILE2%


GOTO END


:USAGE
ECHO ...
ECHO sets Paths in *.copr files
ECHO ...
ECHO Usage: %0
ECHO    [path to be set] -- will be included to result if existing
ECHO    [filenamepattern] -- optional; regular expression depicting 
ECHO       affected files
ECHO    [pathpattern] -- optional; regular expression depicting the
ECHO       paths in file^(s^) to be exchanged
ECHO ...
ECHO notes:
ECHO    - filesearch is always applied recursively
ECHO    - if no filenamepattern is given, all files will be examined
ECHO    - if no pathpattern is given, all Strings beginning with either
ECHO        S'/ or S'C:/ will be exchanged, whereas instead of C any
ECHO        single driveletter is accepted
ECHO ...
ECHO called UnixUtils executables ^(expected to be in PATH^):
ECHO     sed.exe
ECHO     grep.exe
ECHO     tail.exe
ECHO     head.exe
ECHO ...
ECHO ^(C^) Copyright 2009 VISENSO GmbH
ECHO ...
ENDLOCAL
EXIT /B 1


:END


ENDLOCAL

EXIT /B 0