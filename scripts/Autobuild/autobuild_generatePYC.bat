@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM generates binary code of Python scripts
REM *******************************************

SETLOCAL

IF "x%1x" EQU "xx" GOTO USAGE
IF "x%2x" EQU "xx" GOTO USAGE
IF "x%3x" EQU "xx" GOTO USAGE
IF "x%4x" EQU "xx" GOTO USAGE

CALL "%~dp0\..\combinePaths.bat"
CALL common.VISENSO.bat %1 %2
CALL generatePYC.bat %3 %4 %5
GOTO END

:USAGE
ECHO ...
ECHO generates binary code of Python scripts
ECHO ...
ECHO usage:
ECHO %0 
ECHO    [ARCHSUFFIX] <- COVISE architecture suffix
ECHO    [COVISEDIR] <- install path of covise sources (e. g. D:\TRUNK\covise)
ECHO    [path] <- non-optional path containing *.py files to be compiled 
ECHO    [-d ^| -r] <- -p: optional parameter "-q" activates quiet compilation
ECHO                 -r: use python release executable, i e. python.exe 
ECHO    [-q] <- use python debug executable, i. e. python_d.exe; optional
ECHO ...
ECHO called batch scripts and executables:
ECHO     ..\combinePaths.bat
ECHO     common.VISENSO.bat
ECHO     ..\common\generatePYC.bat
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
ENDLOCAL
EXIT /B 1

:END
ENDLOCAL
EXIT /B 0