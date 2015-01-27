@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM ******************************************
REM Creates *.pyc files (which are the binary
REM compilation of *.py files) from *.py files
REM ******************************************
ECHO Will compile all *.py files.
ECHO Starting...

SETLOCAL

IF "x%1x" EQU "xx" GOTO USAGE
REM %2 and %3 are optional

REM TODO: check, if user has permission to write 
REM TODO: check, if enough space is available

SET _PYTHON=python.exe
IF "x%2x" EQU "x-dx" SET _PYTHON=python_d.exe
REM CALL WHERE %_PYTHON% > NUL
REM IF ERRORLEVEL 1 GOTO NOPYTHONCOMPILER

ECHO ...executing %_PYTHON% -v -m compileall -f %3 -x '/\.svn' %1
%_PYTHON% -v -m compileall -f %3 -x '/\.svn' %1

ECHO ...done!
GOTO END

:NOPYTHONCOMPILER
ECHO ...
ECHO No Python compiler %_PYTHON% found!
ECHO Environment not set up or Python installation corrupt.
ECHO ...
GOTO USAGE

:USAGE
ECHO ...
ECHO creates *.pyc files (which are the binary
ECHO compilation of *.py files) from *.py files
ECHO ...
ECHO usage:
ECHO %0 [path] [-d ^| -r] [-q]
ECHO    path: non-optional path containing *.py files to be compiled
ECHO    -p: optional parameter "-q" activates quiet compilation
ECHO    -r: use python release executable, i e. python.exe
ECHO    -d: use python debug executable, i. e. python_d.exe
ECHO ...
ECHO called executables:
ECHO     python.exe [or python_d.exe if -d is given]
ECHO ... 
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
ENDLOCAL
EXIT /B 1

:END
ENDLOCAL
EXIT /B 0