@ECHO OFF

REM ...
REM assembles a Setup.exe for the 64bit RTT distribution of COVISE
REM ...
REM author Harry Trautmann
REM (C) Copyright 2009 VISENSO GmbH
REM ...

SETLOCAL ENABLEDELAYEDEXPANSION

REM -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
REM general distribution variables
SET _ARCHSUFFIX=amdwin64opt
SET _INSTALLTARGET=RTT64
SET COVISE_DISTRO_TYPE=RTT
SET COVISE_DISTRO_TIMEPREFIX=YES
SET COVISE_DISTRO_SKIPSOURCEPREP=

SET COMPUTER=lovell
SET _DIRCOVSRC=C:\COVISE\Covise7.0\covise
SET EXTERNLIBS=C:\EXTERNLIBS
SET _FILEISS=%_DIRCOVSRC%\covise.iss
SET INNOSETUP_HOME=C:\Progra~2\InnoSe~1
SET UNIXUTILS_HOME=%EXTERNLIBS%\UnixUtils
SET GSOAP_HOME=%EXTERNLIBS%\gsoap-2.7
REM -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

ECHO Creating COVISE RTT vista setup on COMPUTER=%COMPUTER%

ECHO ... setting general COVISE variables ...
CALL "%~dp0\..\combinePaths.bat"
CALL common.VISENSO.bat %_ARCHSUFFIX% %_DIRCOVSRC%

REM adapt RTT-specifics
ECHO ... RTT-specific distribution activated ...
COPY "%~dp0\install\RealFluid.bat" %_DIRCOVSRC%\
CALL "%~dp0\..\common\generatePYC.bat" %_DIRCOVSRC%\Python\bin\RTT\Server\Python -r -q
PAUSE

REM actually create installer
CALL "%~dp0\makeCOVISEsetup.bat" %_FILEISS% %_ARCHSUFFIX% %_DIRCOVSRC% %_INSTALLTARGET% %COVISE_DISTRO_TYPE%

ENDLOCAL

PAUSE