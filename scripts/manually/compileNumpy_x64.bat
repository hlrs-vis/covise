@ECHO OFF

REM ***
REM compiles and installes numpy 1.3.0b1 
REM as release and debug builds on x64 architecture
REM ***
REM author: Harry Trautmann
REM (C) 2009 Copyright VISENSO GmbH
REM ***

SETLOCAL ENABLEDELAYEDEXPANSION

ECHO Compiling numpy 1.3.0b on x64...
ECHO ...assure to call this script in Visual Studio x64 command prompt!

REM ***
ECHO This batch is prepared to run on haise.vircinity
SET PYTHONHOME=C:\EXTERNLIBS\Python-2.6.2_x64
REM ***

SET PATH=%PYTHONHOME%\DLLs;%PATH%
ECHO ...PYTHONHOME=%PYTHONHOME%

SET _LOGFILEREL=compileNumpy_release.txt
SET _LOGFILEDBG=compileNumpy_debug.txt

SET _DIRLIBSSRC=build\lib.win-amd64-2.6\numpy
SET _DIRLIBSTRG=%PYTHONHOME%\Lib\site-packages\numpy

REM only for win32, not for x64
REM for WIN32: CALL "C:\Progra~2\Microsoft Visual Studio 8\VC\vcvarsall.bat x86"
REM when using the 64Bit command prompt of VC2005, the 
REM necessary env is already set by
REM for x64: CALL "C:\Progra~2\Microsoft Visual Studio 8\VC\vcvarsall.bat amd64"

SET DISTUTILS_USE_SDK=YES
SET MSSDK=YES

ECHO ...copying python26.lib and python26_d.lib
COPY %PYTHONHOME%\DLLs\Python26.lib %PYTHONHOME%\Lib\
COPY %PYTHONHOME%\DLLs\Python26.lib %PYTHONHOME%\libs\
COPY %PYTHONHOME%\DLLs\Python26_d.lib %PYTHONHOME%\Lib\
COPY %PYTHONHOME%\DLLs\Python26_d.lib %PYTHONHOME%\libs\

ECHO ...running compilation for release build
%PYTHONHOME%\DLLs\Python.exe setup.py install>%_LOGFILEREL%

ECHO ...running compilation for debug build
%PYTHONHOME%\DLLs\Python.exe setup.py build --debug>%_LOGFILEDBG%

ECHO ...copying debug libs to Python installation
COPY %_DIRLIBSSRC%\numarray\*.pyd %_DIRLIBSTRG%\numarray\
COPY %_DIRLIBSSRC%\lib\*.pyd %_DIRLIBSTRG%\lib\
COPY %_DIRLIBSSRC%\core\*.pyd %_DIRLIBSTRG%\core\
COPY %_DIRLIBSSRC%\fft\*.pyd %_DIRLIBSTRG%\fft\
COPY %_DIRLIBSSRC%\linalg\*.pyd %_DIRLIBSTRG%\linalg\
COPY %_DIRLIBSSRC%\random\*.pyd %_DIRLIBSTRG%\random\

ECHO ...done! Please check these logfiles: %_LOGFILEREL% and %_LOGFILEDBG%

ENDLOCAL