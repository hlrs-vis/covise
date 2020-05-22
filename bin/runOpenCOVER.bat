@ECHO OFF
REM this script sets the COVISE env and then starts opencover w/ supplied param
SETLOCAL

CALL c:\src\coviseenv.bat zebuopt

CALL opencover.exe %1 %2 %3 %4 %5

ENDLOCAL