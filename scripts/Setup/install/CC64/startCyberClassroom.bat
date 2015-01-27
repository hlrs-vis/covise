
SETLOCAL ENABLEDELAYEDEXPANSION

REM ########## delete dead memory block remains of previously crashed COVISE calls ##########
DEL /Q "%TMP%\-3690*" > NUL

REM ########## get ARCHSUFFIX of current installation ##########
CALL "%~dp0covise\common.local.bat

SET APPLICATIONDIR=%INSTDIR%\\..
SET DEMODIR=%INSTDIR%\\..\\Demos

cd /d %INSTDIR%\%ARCHSUFFIX%\bin\CyberClassroom\bin
.\Cyber-Classroom.exe "cc" %APPLICATIONDIR% %DEMODIR%

REM # if we get here, delete possible dead memory block remains #
DEL /Q "%TMP%\-3690*" > NUL

ENDLOCAL
