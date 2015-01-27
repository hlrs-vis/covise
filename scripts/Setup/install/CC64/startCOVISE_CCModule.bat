@ECHO ON

REM -----------------------------------------
REM starting Cyberclassroom modules
REM first parameter is used to choose module
REM -----------------------------------------

SETLOCAL

ECHO Starting COVISE Windows XP 64Bit Edition...
ECHO ... (C) Copyright 2009 VISENSO GmbH ...
ECHO ... please be patient ...

REM kill leftover covise processes if powershell is available 
WHERE powershell.exe
IF ERRORLEVEL 0 (
	POWERSHELL.EXE -command stop-process -processname "covise"
	POWERSHELL.EXE -command stop-process -processname "crb"
	POWERSHELL.EXE -command stop-process -processname "OpenCOVER"
)

REM delete dead memory block remains of previously crashed COVISE calls
DEL /Q "%TMP%\-3690*" > NUL

REM get ARCHSUFFIX of current installation
CALL "%~dp0covise\common.local.bat

SET PATH=%PATH%;%INSTDIR%\extern_libs\qt\bin
SET PYTHONPATH=%BASEDIR%
SET APPLICATIONDIR=%INSTDIR%
SET COVISE_HIDDEN=1

REM checking for parameter
IF [%1]==[] ECHO module name parameter missing

SET MODULE=%1
REM SET MODULE=Atombau

REM checking parameter cykloop
SET FILENAME=%2
IF [%2]==[] ECHO file name parameter missing

ECHO MODULE %MODULE%
ECHO FILENAME %FILENAME%

CALL CD /d %COVISEDIR%

REM set up environment variables
SET BASEDIR=%INSTDIR%\..
SET COVISEDIR=%INSTDIR%
SET EXTERNLIBS=%COVISEDIR%\extern_libs
SET ARCHSUFFIX=amdwin64opt
SET COCONFIG=config.%MODULE%.xml
CALL %COVISEDIR%\common.VISENSO.bat %ARCHSUFFIX% %COVISEDIR%

CD /D %COVISEDIR%
CALL runCoviseScriptIF.bat vr-prepare4 %FILENAME% 

:END
ECHO ...done!

ENDLOCAL
