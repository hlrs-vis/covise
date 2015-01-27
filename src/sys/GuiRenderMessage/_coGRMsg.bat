ECHO ON
REM generates the GUI render message interface using SWIG and Qt4

SETLOCAL ENABLEDELAYEDEXPANSION

SET _TEMPDIR=objects_%ARCHSUFFIX%

mkdir %_TEMPDIR%
qmake
%SWIG% -c++ -DGRMSGEXPORT -I%COVISEDIR%\src\kernel\grmsg -I%COVISEDIR%\src\kernel -makedefault -python -o %_TEMPDIR%\coGRMsg_wrap.cpp coGRMsg.i
SET _BUILDTARGET=Debug
if "x%ARCHSUFFIX:~-3,3%x" EQU "xoptx" SET _BUILDTARGET=Release
devenv _coGRMsg_%ARCHSUFFIX%.vcproj /rebuild !_BUILDTARGET!

move %_TEMPDIR%\coGRMsg.py %COVISEDIR%\%ARCHSUFFIX%\lib\coGRMsg.py
move %COVISEDIR%\%ARCHSUFFIX%\lib\_coGRMsg.dll %COVISEDIR%\%ARCHSUFFIX%\lib\_coGRMsg.pyd

ENDLOCAL