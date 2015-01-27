@ECHO OFF

IF "x%1x" EQU "xx" GOTO ERROR
ECHO Starting Microsoft Visual Studio 2008 Express for ARCHSUFFIX=%1

SETLOCAL

SET COVISEDIR="%~dp0..\.."

ECHO COVISEDIR=%COVISEDIR%

CALL "%~dp0\..\combinePaths.bat"
CALL common.VISENSO.bat %1 %COVISEDIR%

REM setup Visual C++ environment
REM default to 32bit, but check for 64bit targets
SET _TARGET=x86
IF "x%1:~0,-3%x" EQU "xamdwin64x" SET _TARGET=amd64
IF "x%1:~0,-3%x" EQU "xangusx" SET _TARGET=amd64

IF _TARGET EQU "x86" (
   CALL vcvarsall.bat %_TARGET%
   CALL devenv
) ELSE (
   "C:\Program Files (x86)\Microsoft Visual Studio 9.0\Common7\IDE\VCExpress.exe"
)

ENDLOCAL

GOTO END

:ERROR
ECHO ERROR: No ARCHSUFFIX supplied as parameter 1
ECHO USAGE: %0 [ARCHSUFFIX]
GOTO END

:END