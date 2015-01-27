@ECHO OFF

IF "x%1x" EQU "xx" GOTO ERROR
ECHO Starting Microsoft Visual Studio 2005 for ARCHSUFFIX=%1

SETLOCAL

SET COVISEDIR="%~dp0..\.."

ECHO COVISEDIR=%COVISEDIR%

CALL "%~dp0\..\combinePaths.bat"
CALL common.VISENSO.bat %1 %COVISEDIR%

REM setup Visual C++ environment
REM default to 32bit, but check for 64bit targets
SET _TARGET=x86
IF %ARCHSUFFIX% EQU amdwin64 SET _TARGET=amd64
IF %ARCHSUFFIX% EQU amdwin64opt SET _TARGET=amd64
IF %ARCHSUFFIX% EQU angus SET _TARGET=amd64
IF %ARCHSUFFIX% EQU angusopt SET _TARGET=amd64

IF %_TARGET% EQU x86 (
   "C:\Progra~1\Microsoft Visual Studio 8\Common7\IDE\devenv.exe"
) ELSE (
   "C:\Program Files (x86)\Microsoft Visual Studio 8\Common7\IDE\devenv.exe"
)

ENDLOCAL

GOTO END

:ERROR
ECHO ERROR: No ARCHSUFFIX supplied as parameter 1
ECHO USAGE: %0 [ARCHSUFFIX]
GOTO END

:END