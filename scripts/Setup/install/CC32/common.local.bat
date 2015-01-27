@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM Sets the local paths of applications needed
REM to either compile or run COVISE that are 
REM outside the EXTERNLIBS directory or differ
REM of the default path
REM *******************************************

SET PYTHONVERSION=26
SET ARCHSUFFIX=vistaopt

REM get the 8.3 filename of current installation
SET TMPFILE="%TMP%\~covtmpfile.txt"
CALL "%WINDIR%\system32\cscript.exe" /nologo "%~dp0get8dot3path.vbs" "%~dp0">%TMPFILE%
SET /P INSTDIR=<%TMPFILE%
DEL /Q %TMPFILE%

SET EXTERNLIBS=%INSTDIR%\extern_libs