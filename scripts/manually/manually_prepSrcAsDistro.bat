@ECHO OFF

REM ***
REM manual entry point for routine that
REM prepares the COVISE sources as distribution,
REM which means that COVISE can be started out of the source directories
REM after preparation
REM ***
REM author: Harry Trautmann
REM (C) 2009 Copyright VISENSO GmbH
REM ***

SETLOCAL ENABLEDELAYEDEXPANSION

REM **********
REM POSSIBLY ADAPT THESE TO LOCAL COMPUTER ENVIRONMENT
REM **********
SET COMPUTER=o2
SET COVISEDIR=%~dp0..\..
SET ARCHSUFFIX=vistaopt
SET EXTERNLIBS=c:\vista
SET UNIXUTILS=c:\vista\UnixUtils
REM **********

ECHO COMPUTER=%_COMPUTER%
CALL "%~dp0..\common\prepSrcAsDistro.bat" %COVISEDIR% %ARCHSUFFIX% %EXTERNLIBS% %UNIXUTILS%

ENDLOCAL