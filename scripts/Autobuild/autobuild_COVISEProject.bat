@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM sets up the environment for building a 
REM given Microsoft Visual C++ solution
REM *******************************************

SETLOCAL

IF "x%1x" EQU "xx" GOTO USAGE
IF "x%2x" EQU "xx" GOTO USAGE
IF "x%3x" EQU "xx" GOTO USAGE
IF "x%4x" EQU "xx" GOTO USAGE
IF "x%5x" EQU "xx" GOTO USAGE

CALL "%~dp0\..\combinePaths.bat"
CALL common.VISENSO.bat %1 %2

REM setup Visual C++ environment
REM default to 32bit, but check for 64bit targets
SET _TARGET=x86
IF "x%1:~0,-3%x" EQU "xamdwin64x" SET _TARGET=amd64
IF "x%1:~0,-3%x" EQU "xangusx" SET _TARGET=amd64
CALL vcvarsall.bat %_TARGET%

CD /D %2\src
SET _ARCHSUFFIX=%1

SET _VCBUILDTYPE=Debug
IF "x%_ARCHSUFFIX:~-3,3%x" EQU "xoptx" SET _VCBUILDTYPE=Release

devenv.com "%2\src\%3" /rebuild "%_VCBUILDTYPE%">"%5\results_%4.txt"

GOTO END

:USAGE
ECHO ...
ECHO sets up the environment for building a 
ECHO given Microsoft Visual C++ solution
ECHO ...
ECHO usage:
ECHO %0 
ECHO    [ARCHSUFFIX] <- COVISE architecture suffix
ECHO    [COVISEDIR] <- install path of covise sources (e. g. D:\TRUNK\covise)
ECHO    [path + solutionname] <- path is to be relative to %COVISEDIR%\src
ECHO       (e. g. renderer\OpenCOVER\OpenCOVER_vistaopt.sln)
ECHO    [reportfile suffix] <- e. g. OpenCOVER
ECHO    [path of reportfile] <- e. g. C:\temp\reports
ECHO ...
ECHO called batch scripts and executables:
ECHO     common.VISENSO.bat
ECHO     vcvarsall.bat
ECHO     devenv.com
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
ENDLOCAL
EXIT /B 1

:END
ENDLOCAL
EXIT /B 0