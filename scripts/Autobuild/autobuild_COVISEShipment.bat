@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM sets up directory structure containing 
REM COVISE files in the right order
REM *******************************************

SETLOCAL

IF /I "x%1x" EQU "xx" GOTO USAGE
IF /I "x%2x" EQU "xx" GOTO USAGE
IF /I "x%3x" EQU "xx" GOTO USAGE
IF /I "x%4x" EQU "xx" GOTO USAGE
IF /I "x%5x" EQU "xx" GOTO USAGE
IF /I "x%6x" EQU "xx" GOTO USAGE

SET COVSRC=%1
SET COVINSTALL=%2
SET ARCHSUFFIX=%3
SET LICENSE=%4
SET UNIXUTILS=%5
SET INSTALLTARGET=%6

CALL "%~dp0\..\combinePaths.bat"
CALL common.VISENSO.bat %ARCHSUFFIX% %COVSRC%
CALL makeCOVISEShipment.bat %COVSRC% %COVINSTALL% %LICENSE% %UNIXUTILS% %INSTALLTARGET%

GOTO END

:USAGE
ECHO ...
ECHO usage: 
ECHO %0 
ECHO    [path to COVISE source files]
ECHO    [path to COVISE installation destination directory]
ECHO    [COVISE architecture suffix]
ECHO    [path to config.license.xml of this shipment]
ECHO    [path to unix utils, like grep, sed and head]
ECHO    [folder name containing files specific to distribution]
ECHO ...
ECHO note: if no path to license file given, the existing one is maintained;
echo ...
ECHO called batch scripts:
ECHO     ..\combinePaths.bat
ECHO     common.VISENSO.bat
ECHO     makeCOVISEShipment.bat
ECHO ...
ENDLOCAL
EXIT /B 1

:END
ENDLOCAL
EXIT /B 0