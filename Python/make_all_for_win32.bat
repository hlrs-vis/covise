
@ echo off

SETLOCAL ENABLEDELAYEDEXPANSION

REM #########################################################################
REM # this file substitute the Makefile for windows
REM #########################################################################

rem copy python startup script to the bin directory
rem this script is started by the crb

copy scriptInterface.bat  %COVISEDIR%\%ARCHSUFFIX%\bin\scriptInterface.bat

echo scriptInterface.bat copied to %COVISEDIR%\%ARCHSUFFIX%\bin


rem generate the static python representation of all covise module
rem find errors in the stub file
rem some modules can't be converted under windows 

echo will create stubs for all modules
echo not all will have success, be patient ....

SET _PYTHON=%PYTHON_HOME%\bin\python.exe
IF DEFINED COVISE_LOCAL_PYTHON (
   ECHO using local python interpreter %COVISE_LOCAL_PYTHON%
   SET _PYTHON=%COVISE_LOCAL_PYTHON%
)

SET _MAKEBASIIGNORELIST=
IF EXIST %COVISEDIR%\Python\makeBasiModIgnorelist.txt (
   FOR /F %%G IN (%COVISEDIR%\Python\makeBasiModIgnorelist.txt) DO (
      SET _MAKEBASIIGNORELIST=!_MAKEBASIIGNORELIST! -i%%G
   )
)
ECHO %_PYTHON% %COVISEDIR%\Python\makeBasiModCode.py %_MAKEBASIIGNORELIST%
%_PYTHON% %COVISEDIR%\Python\makeBasiModCode.py %_MAKEBASIIGNORELIST% > coPyModules.py
echo finished !!

ENDLOCAL