@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM wraps the call to the batch, that creates
REM a setup.exe for COVISE
REM *******************************************

SETLOCAL

IF /I "x%1x" EQU "xx" GOTO USAGE
IF /I "x%2x" EQU "xx" GOTO USAGE
IF /I "x%3x" EQU "xx" GOTO USAGE
IF /I "x%4x" EQU "xx" GOTO USAGE

SET SETUPSCRIPT=%1
SET ARCHSUFFIX=%2
SET COVSRC=%3
SET QIET=%4

CALL "%~dp0\..\combinePaths.bat"
CALL common.VISENSO.bat %ARCHSUFFIX% %COVSRC%
CALL makeCOVISESetup.bat %SETUPSCRIPT% %ARCHSUFFIX% %COVSRC% %QIET%

GOTO END

:USAGE
ECHO ...
ECHO usage: 
ECHO %0 
ECHO    [absolute path + filename of setup script]
ECHO    [target COVISE archsuffix]
ECHO    [path of COVISE sources]
ECHO    [supply /Q for silent compile run]
ECHO ...
ECHO called batch scripts:
ECHO     ..\combinePaths.bat
ECHO     common.VISENSO.bat
ECHO     ..\Setup\makeCOVISEsetup.bat
ECHO ...
ENDLOCAL
EXIT /B 1

:END
ENDLOCAL
EXIT /B 0