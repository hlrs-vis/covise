@echo off

if defined COMMON_ACTIVE (
   goto END
)

if "%1" == "--help" (
   echo common.bat [ARCHSUFFIX]
   echo "ARCHSUFFIX: win32, win32opt, amdwin64, amdwin64opt, ia64win, vista (default), vistaopt, zackel, zackelopt, angus, angusopt, yoroo, yorooopt, berrenda, berrendaopt, tamarau, tamarauopt, mingw, mingwopt"
   pause
   goto END
)

if "%1" == "" (
  if not defined ARCHSUFFIX% set ARCHSUFFIX=vista
) else (
  set ARCHSUFFIX=%1
)

if not defined COVISE_BRANCH (
   set COVISE_BRANCH=HLRS
)

if not defined EXTERNLIBS (
   if not defined EXTERNLIBSROOT (
      echo EXTERNLIBS and EXTERNLIBSROOT are not set
      pause
      goto END
   ) else (
      set EXTERNLIBS=%EXTERNLIBSROOT%\%ARCHSUFFIX%
   )
)


if defined COVISEDIR goto COVISEDIR_IS_OK
REM to avoid problems with closing rounded bracket in IF-clause
REM work with GOTOs to set missing COVISEDIR

REM current working path looks like COVISEDIR
if exist "%CD%"\common.bat echo setting COVISEDIR to %CD%
if exist "%CD%"\common.bat set COVISEDIR=%CD%
if exist "%CD%"\common.bat goto COVISEDIR_IS_OK

REM see, if path where this script is called looks like COVISEDIR
REM accept path if common-base.bat seems to be at the right place
if exist "%~dp0"\bin\common-base.bat echo setting COVISEDIR to %~dp0
if exist "%~dp0"\bin\common-base.bat set COVISEDIR=%~dp0
if exist "%~dp0"\bin\common-base.bat goto COVISEDIR_IS_OK

REM else no suitable COVISEDIR found; abort
echo COVISEDIR is not set
pause
goto END

:COVISEDIR_IS_OK

if defined CADCV_DIR (
   cd /d %CADCV_DIR%/qmakehelp
   call qmakehelp.bat
   cd /d %COVISEDIR%
)

if not defined COFRAMEWORKDIR (
   set COFRAMEWORKDIR=%COVISEDIR%
)

if /I "%ARCHSUFFIX%" EQU "amdwin64opt" goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "amdwin64" goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "ia64win"  goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "win32opt" goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "win32"    goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "vistaopt" goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "vista"    goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "zackelopt" goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "zackel"    goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "angusopt" goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "angus"    goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "yorooopt" goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "yoroo"    goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "berrendaopt" goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "berrenda"    goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "tamarauopt" goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "tamarau"    goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "mingwopt" goto LABEL_SETENVIRONMENT
if /I "%ARCHSUFFIX%" EQU "mingw"    goto LABEL_SETENVIRONMENT
echo ARCHSUFFIX %ARCHSUFFIX% is not supported!
echo common.bat [ARCHSUFFIX]
echo "ARCHSUFFIX: win32, win32opt, amdwin64, amdwin64opt, ia64win, vista (default), vistaopt, zackel, zackelopt, angus, angusopt, yoroo, yorooopt, berrenda, berrendaopt, tamarau, tamarauopt, mingw, mingwopt"
pause
goto END



:LABEL_SETENVIRONMENT
echo Environment settings for ARCHSUFFIX %ARCHSUFFIX%

set BASEARCHSUFFIX=%ARCHSUFFIX:opt=%

cd /d %COVISEDIR%
         
if exist "%COVISEDIR%"\mycommon.bat (
   call "%COVISEDIR%"\mycommon.bat
   echo mycommon.bat was included
)

if /I "%BASEARCHSUFFIX%" EQU "mingw" (
  set COMPILE_WITH_MAKE=1
  call "%COVISEDIR%\bin\common-base-mingw.bat"
) else (
  call "%COVISEDIR%\bin\common-base.bat"
)

set FRAMEWORK=covise
set QMAKECOVISEDIR=%COVISEDIR%
set LOGNAME=covise
set PATH=%PATH%;%COVISEDIR%\%ARCHSUFFIX%\bin;%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\bin;%COVISEDIR%\%ARCHSUFFIX%\bin\Renderer;%COVISEDIR%\%ARCHSUFFIX%\lib\opencover\plugins
rem no longer used
rem set PATH=%PATH%;%EXTERNLIBS%\wget\bin;%EXTERNLIBS%\molscript\bin
rem no longer neededset QMAKESPEC=%COVISEDIR%\mkspecs\%ARCHSUFFIX%

if not defined COVISEDESTDIR   set COVISEDESTDIR=%COVISEDIR%
if not defined VV_SHADER_PATH  set VV_SHADER_PATH=%COVISEDIR%\src\3rdparty\deskvox\virvo\shader
if not defined COVISE_PATH     set COVISE_PATH=%COVISEDIR%


set RM=rmdir /S /Q
set BISON=%COVISEDIR%\bin\bison.exe
set BISONPLUSPLUS=%COVISEDIR%\bin\bison++.exe
set LEX=%COVISEDIR%\bin\flex.exe
set YACC=%COVISEDIR%\bin\yacc
if "%ARCHSUFFIX%" EQU "vista"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "zackel"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "angus"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "yoroo"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "berrenda"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "tamarau"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "mingw"  set COVISE_DEVELOPMENT=YES

set COMMON_ACTIVE=1
:END
