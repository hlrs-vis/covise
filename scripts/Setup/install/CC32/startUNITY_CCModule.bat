@ECHO ON

REM -----------------------------------------
REM starting Unity Cyberclassroom modules
REM first parameter is used to choose module  
REM -----------------------------------------

SETLOCAL

ECHO ... (C) Copyright 2013 VISENSO GmbH ...
ECHO ... please be patient ...

REM get ARCHSUFFIX of current installation
CALL "%~dp0covise\common.local.bat

REM checking for parameter
IF [%1]==[] ECHO exe parameter missing

REM checking parameter cykloop
IF [%2]==[] ECHO command line flags missing 

REM set up environment variables
SET COVISEDIR=%INSTDIR%
SET UNITYVR_CONFIG_DIR=%COVISEDIR%\config\

CALL %*

:END
ECHO ...done!

ENDLOCAL
