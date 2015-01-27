@echo off

rem  ---------------------------------------------
rem  clean_covise.bat <parameter>
rem
rem  parameter: [off|on] quiet mode on/off
rem                      default is on
rem  ---------------------------------------------

echo clean running covise environment...

if /I "%FRAMEWORK%" == "covise" set WORKINGDIR=%COVISEDIR%
if /I "%FRAMEWORK%" == "yac"    set WORKINGDIR=%YACDIR%


rem  ---------------------------------------------
rem  check QUIET mode
rem  ---------------------------------------------
set QUIET=%1


if /I "%QUIET%" EQU "OFF" goto QUIET_CHECK_OK
if /I "%QUIET%" EQU "ON"  goto QUIET_CHECK_OK
set QUIET=ON


:QUIET_CHECK_OK


rem  ---------------------------------------------
rem  call individualized CleanCovise.bat
rem  ---------------------------------------------
if exist %WORKINGDIR%\%ARCHSUFFIX%\bin\CleanCoviseLocal.bat (
   call %WORKINGDIR%\%ARCHSUFFIX%\bin\CleanCoviseLocal.bat %QUIET%
   goto CLEANED
)


rem  ---------------------------------------------
rem  clean remaining covise processes
rem  ---------------------------------------------
if exist %SYSTEMROOT%\system32\taskkill.exe (
   if /I %QUIET% EQU OFF echo ...clean remaining covise processes
   echo ...first
   tasklist /NH > tasklist.log
   FOR /F %%v IN ( tasklist.log ) DO (
rem      if "%%v" EQU "covise.exe"      taskkill /T /f /im covise.exe
   )
   echo ...second
   tasklist /NH /FI "MODULES eq coCore.dll" > tasklist.log
   FOR /F "tokens=1,2" %%v IN ( tasklist.log ) DO (
      echo ...%%v killed.
      taskkill /f /im %%w 
   )
   del tasklist.log
   if /I "%QUIET%" EQU "OFF" echo ...done.
) else (
   if /I "%QUIET%" EQU "OFF" echo Your operating system doesnot support command based task killing.
   if /I"%QUIET%" EQU "OFF" echo Thus, typical covise processes cannot be killed by this batch file.
)


rem  ---------------------------------------------
rem  clean remaining shm segments
rem  ---------------------------------------------
if /I "%QUIET%" EQU "OFF" echo ...clean remaining shm segments
if exist "%USERPROFILE%\Local Settings\Temp" cd %USERPROFILE%\"Local Settings\Temp"
if exist "%USERPROFILE%\Lokale Einstellungen\Temp" cd %USERPROFILE%\"Lokale Einstellungen\Temp"
if exist -* (
   dir /B /S -* > shm.log
   if exist shm.log (
      for /F "tokens=*" %%v in (shm.log) do (
         if /I "%QUIET%" EQU "OFF" echo %%v
         del /F /Q "%%v"
      )
      del shm.log
   )
)
if /I "%QUIET%" EQU "OFF" echo ...done.

rem  ---------------------------------------------
rem  end
rem  ---------------------------------------------
cd %WORKINGDIR%
echo ...all cleaned.
:CLEANED
echo on
