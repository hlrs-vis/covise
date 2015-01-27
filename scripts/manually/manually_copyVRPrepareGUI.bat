@ECHO OFF

SETLOCAL ENABLEDELAYEDEXPANSION

IF "x%1x" EQU "xx" GOTO USAGE
IF "x%2x" EQU "xx" GOTO USAGE

CALL "%~dp0\..\combinePaths.bat"

ECHO Source: %1
ECHO Target: %2
ECHO Will mostly copy the *.pyc files; whenever a *.py file misses, edit me!
ECHO Copying ...

REM get the 8.3 filename of source and target paths
SET TMPFILE="%TMP%\~covtmpfile.txt"
CALL "%WINDIR%\system32\cscript.exe" /nologo get8dot3path.vbs "%1">%TMPFILE%
SET /P SRCDIR=<%TMPFILE%
DEL /Q %TMPFILE%
CALL "%WINDIR%\system32\cscript.exe" /nologo get8dot3path.vbs "%2">%TMPFILE%
SET /P DSTDIR=<%TMPFILE%
DEL /Q %TMPFILE%

COPY /V /Y %SRCDIR%\Python\bin\vr-prepare\*.pyc %DSTDIR%\Python\bin\vr-prepare\
COPY /V /Y %SRCDIR%\Python\bin\vr-prepare\vr-prepare.pyc %DSTDIR%\Python\bin\vr-prepare\
COPY /V /Y %SRCDIR%\Python\bin\vr-prepare\converters\*.pyc %DSTDIR%\Python\bin\vr-prepare\converters\
COPY /V /Y %SRCDIR%\Python\bin\vr-prepare\negotiator\*.pyc %DSTDIR%\Python\bin\vr-prepare\negotiator\
COPY /V /Y %SRCDIR%\Python\bin\vr-prepare\negotiator\unittests\*.pyc %DSTDIR%\Python\bin\vr-prepare\negotiator\unitests\
COPY /V /Y %SRCDIR%\Python\bin\vr-prepare\negotiator\import\unittests\*.pyc %DSTDIR%\Python\bin\vr-prepare\negotiator\import\unitests\

ECHO ...done!

GOTO END

:USAGE
ECHO Usage:
ECHO %0
ECHO    [source COVISE base directory, e. g. C:\COVISE\covise]
ECHO    [target COVISE base directory, e. g. C:\Progra~1\COVISE\covise]
ECHO ...
ECHO copies UI Python classes of COVISE VR-Prepare GUI from a source directory
ECHO to a target directory (e. g. where COVISE is installed)
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
GOTO END

:END

ENDLOCAL