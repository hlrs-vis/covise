@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM calls qmake to generate Microsoft Visual
REM C++ Projects
REM *******************************************

SETLOCAL
IF "x%1x" EQU "xx" GOTO USAGE
IF "x%2x" EQU "xx" GOTO USAGE
IF "x%3x" EQU "xx" GOTO USAGE

CALL "%~dp0\..\combinePaths.bat"
CALL common.VISENSO.bat %1 %2
CD /D %3
qmake.exe -r
GOTO END

:USAGE
ECHO ...
ECHO calls qmake to generate Microsoft Visual
ECHO C++ Projects
ECHO ...
ECHO usage:
ECHO %0 
ECHO    [ARCHSUFFIX]
ECHO    [COVISEDIR]
ECHO    [directory to qmake]
ECHO ...
ECHO called batch scripts and executables:
ECHO     ..\combinePaths.bat
ECHO     common.VISENSO.bat
ECHO     %QTDIR%\bin\qmake.exe
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
ENDLOCAL
EXIT /B 1

:END
ENDLOCAL
EXIT /B 0