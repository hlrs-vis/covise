@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM builds the COVISE GuiRenderMessage module
REM *******************************************

SETLOCAL

IF "x%1x" EQU "xx" GOTO USAGE
IF "x%2x" EQU "xx" GOTO USAGE
IF "x%3x" EQU "xx" GOTO USAGE

CALL "%~dp0\..\combinePaths.bat"
CALL common.VISENSO.bat %1 %2

REM setup Visual C++ environment
REM default to 32bit, but check for 64bit targets
SET _TARGET=x86
IF "x%1:~0,-3%x" EQU "xamdwin64x" SET _TARGET=amd64
IF "x%1:~0,-3%x" EQU "xangusx" SET _TARGET=amd64
CALL vcvarsall.bat %_TARGET%

CD /D %2\src\sys\GuiRenderMessage
CALL _coGRMsg.bat > %3\results_coGRMsg.txt
GOTO END

:USAGE
ECHO ...
ECHO builds the COVISE GuiRenderMessage module
ECHO ...
ECHO usage:
ECHO %0 
ECHO    [ARCHSUFFIX] <- COVISE architecture suffix
ECHO    [COVISEDIR] <- install path of covise sources (e. g. D:\TRUNK\covise)
ECHO    [path of reportfile] <- e. g. C:\temp\reports
ECHO ...
ECHO called batch scripts:
ECHO     common.VISENSO.bat
ECHO     vcvarsall.bat
ECHO     _coGRMsg.bat
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
ENDLOCAL
EXIT /B 1

:END
ENDLOCAL
EXIT /B 0