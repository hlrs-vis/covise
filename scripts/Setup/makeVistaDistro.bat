@ECHO OFF

REM ...
REM assembles a Setup.exe for the Vista distribution of COVISE
REM ...
REM author Harry Trautmann
REM (C) Copyright 2009 VISENSO GmbH
REM ...

SETLOCAL ENABLEDELAYEDEXPANSION

REM -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
REM general distribution variables
SET _ARCHSUFFIX=vistaopt
SET _INSTALLTARGET=vista
SET COVISE_DISTRO_TYPE=PLAINVANILLA
SET COVISE_DISTRO_TIMEPREFIX=YES
SET COVISE_DISTRO_SKIPSOURCEPREP=

SET COMPUTER=o2.vircinity
SET _DIRCOVSRC=w:\ko_te\Covise\branches\Covise7.0\covise
SET EXTERNLIBS=C:\vista
SET _FILEISS=%_DIRCOVSRC%\covise.iss
SET INNOSETUP_HOME=C:\Programme\Inno Setup 5
SET UNIXUTILS_HOME=%EXTERNLIBS%\UnixUtils
SET GSOAP_HOME=%EXTERNLIBS%\gsoap
REM -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

ECHO Creating plain vanilla COVISE vista setup on COMPUTER=%COMPUTER%

REM actually create installer
CALL "%~dp0makeCOVISEsetup.bat" %_FILEISS% %_ARCHSUFFIX% %_DIRCOVSRC% %_INSTALLTARGET% %COVISE_DISTRO_TYPE%

ENDLOCAL

PAUSE