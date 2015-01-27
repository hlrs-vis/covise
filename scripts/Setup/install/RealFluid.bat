REM @ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM Startup of the COVISE RealFluid Server
REM *******************************************

SETLOCAL ENABLEDELAYEDEXPANSION

DEL /Q "%TMP%\-3690*" > NUL

REM get the 8.3 filename of current installation
SET TMPFILE="%TMP%\~covtmpfile.txt"
CALL "%WINDIR%\system32\cscript.exe" /nologo "%~dp0get8dot3path.vbs" "%~dp0">%TMPFILE%
SET /P INSTDIR=<%TMPFILE%
DEL /Q %TMPFILE%

SET EXTERNLIBS=%INSTDIR%\extern_libs

REM note: ARCHSUFFIX has to be set in common.local.bat
CALL %INSTDIR%\common.VISENSO.bat NONE %INSTDIR%

CALL %INSTDIR%\runCoviseScriptIf.bat RealFluid %1

DEL /Q "%TMP%\-3690*" > NUL

ENDLOCAL