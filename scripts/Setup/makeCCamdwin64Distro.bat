@ECHO OFF

REM ...
REM assembles a Setup.exe for the Cyberclassroom vista distribution of COVISE
REM ...
REM author Harry Trautmann
REM (C) Copyright 2009 VISENSO GmbH
REM ...

SETLOCAL ENABLEDELAYEDEXPANSION

REM -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
REM general distribution variables
SET _ARCHSUFFIX=amdwin64opt
SET _INSTALLTARGET=CC64
SET COVISE_DISTRO_TYPE=CC
SET COVISE_DISTRO_TIMEPREFIX=YES
SET COVISE_DISTRO_SKIPSOURCEPREP=
SET CC_SCHOOL_EDITION=YES
SET CC_VOCATIONAL_EDITION=YES

SET COMPUTER=CC-32-INST
SET _DIRCOVSRC=D:\Covise7.0\covise
SET EXTERNLIBS=D:\EXTERNLIBS
SET DEMO_DIR=D:\CC-Modules
SET VISENSO_BRANCH=D:\Visenso
SET PROJECT_DIR=Y:\common\Projekte\VR4Schule
SET _FILEISS=%_DIRCOVSRC%\covise.iss
SET INNOSETUP_HOME=C:\Progra~2\Inno Setup 5
SET UNIXUTILS_HOME=%EXTERNLIBS%\UnixUtils
SET GSOAP_HOME=%EXTERNLIBS%\gsoap-2.7
REM -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

ECHO Creating COVISE CyberClassroom setup on COMPUTER=%COMPUTER%

REM adapt CC-specifics
ECHO ... CC-specific distribution activated ...
COPY "%~dp0\install\%_INSTALLTARGET%\startCyberClassroom.bat" %_DIRCOVSRC%\..\
COPY "%~dp0\install\%_INSTALLTARGET%\startCOVISE_CCModule.bat" %_DIRCOVSRC%\
COPY "%~dp0\install\%_INSTALLTARGET%\startUNITY_CCModule.bat" %_DIRCOVSRC%\
COPY "%~dp0\install\%_INSTALLTARGET%\startTracking.bat" %_DIRCOVSRC%\
PAUSE

REM actually create installer
CALL "%~dp0makeCOVISEsetup.bat" %_FILEISS% %_ARCHSUFFIX% %_DIRCOVSRC% %_INSTALLTARGET% %COVISE_DISTRO_TYPE%

ENDLOCAL

PAUSE
