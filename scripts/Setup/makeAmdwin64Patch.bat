@ECHO OFF

REM ...
REM assembles a Setup.exe for the amdwin64 distribution of COVISE
REM ...
REM author Harry Trautmann
REM (C) Copyright 2009 VISENSO GmbH
REM ...

SETLOCAL ENABLEDELAYEDEXPANSION

REM -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
REM general distribution variables
SET _ARCHSUFFIX=amdwin64opt
SET _INSTALLTARGET=amdwin64
SET COVISE_DISTRO_TYPE=PLAINVANILLA
SET COVISE_DISTRO_TIMEPREFIX=YES
SET COVISE_DISTRO_SKIPSOURCEPREP=

SET COMPUTER=haise.vircinity
SET _DIRCOVSRC=C:\COVISE\Autobuild\amdwin64\Covise7.0\covise
SET EXTERNLIBS=C:\EXTERNLIBS
SET _FILEISS=%_DIRCOVSRC%\patch.iss
SET INNOSETUP_HOME=C:\Progra~2\Inno Setup 5
SET UNIXUTILS_HOME=%EXTERNLIBS%\UnixUtils
SET GSOAP_HOME=%EXTERNLIBS%\gsoap-2.7
REM -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

ECHO Creating plain vanilla COVISE amdwin64 setup on COMPUTER=%COMPUTER%

REM actually create installer
CALL "%~dp0\makeCOVISEsetup.bat" %_FILEISS% %_ARCHSUFFIX% %_DIRCOVSRC% %_INSTALLTARGET% %COVISE_DISTRO_TYPE%

ENDLOCAL

PAUSE