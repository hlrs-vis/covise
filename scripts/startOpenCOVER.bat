@echo off
set ARCHSUFFIX=%1
shift
set COCONFIG=config_ATS_left.xml
shift
echo COCONFIG %COCONFIG%
rem echo %*
echo hallo
echo %1
echo hallo2
start /b opencover %1 %2 %3 %4 %5 %6 %7 %8 %9