@ECHO OFF

REM ...
REM assembles a Setup.exe for the 32bit RTT distribution of COVISE
REM ...
REM author Harry Trautmann
REM (C) Copyright 2009 VISENSO GmbH
REM ...

SETLOCAL ENABLEDELAYEDEXPANSION

REM -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
REM general distribution variables
SET _ARCHSUFFIX=vistaopt
SET _INSTALLTARGET=RTT32
SET COVISE_DISTRO_TYPE=RTT
SET COVISE_DISTRO_TIMEPREFIX=YES
SET COVISE_DISTRO_SKIPSOURCEPREP=

SET COMPUTER=o2.vircinity
SET _DIRCOVSRC=D:\COVISE\Autobuild\vista\COVISE_nightly_src\covise
SET EXTERNLIBS=C:\vista
SET _FILEISS=%_DIRCOVSRC%\covise.iss
SET INNOSETUP_HOME=C:\Programme\Inno Setup 5
SET UNIXUTILS_HOME=%EXTERNLIBS%\UnixUtils
SET GSOAP_HOME=%EXTERNLIBS%\gsoap
REM -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

ECHO Creating COVISE RTT vista setup on COMPUTER=%COMPUTER%

ECHO ... setting general COVISE variables ...
CALL "%~dp0\..\combinePaths.bat"
CALL common.VISENSO.bat %_ARCHSUFFIX% %_DIRCOVSRC%

REM adapt RTT-specifics
ECHO ... RTT-specific distribution activated ...
COPY "%~dp0\install\RealFluid.bat" %_DIRCOVSRC%\
IF EXIST %_DIRCOVSRC%\Python\bin\RTT RMDIR /S /Q %_DIRCOVSRC%\Python\bin\RTT
XCOPY /S /Y %_DIRCOVSRC%\src\Visenso\Http-Vis-Server\RTT %_DIRCOVSRC%\Python\bin\RTT\
CALL "%~dp0\..\common\generatePYC.bat" %_DIRCOVSRC%\Python\bin\RTT\Server\Python -r -q
PAUSE

REM actually create installer
CALL "%~dp0\makeCOVISEsetup.bat" %_FILEISS% %_ARCHSUFFIX% %_DIRCOVSRC% %_INSTALLTARGET% %COVISE_DISTRO_TYPE%

ENDLOCAL

PAUSE