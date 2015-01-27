@ECHO OFF

REM ************************************************************
REM 
REM This script starts VRPrepare4 GUI.
REM
REM Usage: 
REM    runVRPrepare4.bat 
REM       <name of python script>  
REM       <script argument 1> 
REM       ... 
REM       <script argument n>
REM
REM (C) Copyright 2009 VISENSO GmbH
REM
REM ************************************************************

REM changes to environment variables are only local
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

CALL "%~dp0\runCoviseScriptIF.bat" vr-prepare4 %*

DEL /Q "%TMP%\-3690*" > NUL

ENDLOCAL
EXIT /B 0