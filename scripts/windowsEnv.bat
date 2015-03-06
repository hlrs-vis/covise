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
  if not defined ARCHSUFFIX% set ARCHSUFFIX=tamarauopt
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
      set "EXTERNLIBS=%EXTERNLIBSROOT%\%ARCHSUFFIX%"
   )
)


if defined COVISEDIR goto COVISEDIR_IS_OK
REM to avoid problems with closing rounded bracket in IF-clause
REM work with GOTOs to set missing COVISEDIR

REM current working path looks like COVISEDIR
if exist "%CD%"\windowsEnv.bat echo setting COVISEDIR to %CD%
if exist "%CD%"\windowsEnv.bat set COVISEDIR=%CD%
if exist "%CD%"\windowsEnv.bat goto COVISEDIR_IS_OK

REM see, if path where this script is called looks like COVISEDIR
REM accept path if common-base.bat seems to be at the right place
if exist "%~dp0"\windowsEnv.bat echo setting COVISEDIR to %~dp0
if exist "%~dp0"\windowsEnv.bat set COVISEDIR=%~dp0
if exist "%~dp0"\windowsEnv.bat goto COVISEDIR_IS_OK

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

cd /d "%COVISEDIR%"
if exist "%COVISEDIR%"\mysetenv.bat (
   call "%COVISEDIR%"\mysetenv.bat
   echo mysetenv.bat was included
)


if not defined QT_HOME ( 
   REM QT_HOME is not set... check QTDIR
   IF not defined QTDIR (
     REM QTDIR is not set ! Try in EXTERNLIBS
     set "QT_HOME=%EXTERNLIBS%\qt5"
     set "QT_SHAREDHOME=%EXTERNLIBS%\qt5"
     set "QTDIR=%EXTERNLIBS%\qt5"
     set "QT_INCPATH=%EXTERNLIBS%\qt5\include"
     set "QT_LIBPATH=%EXTERNLIBS%\qt5\lib"
	 set "PATH=%EXTERNLIBS%\qt5\bin;%EXTERNLIBS%\qt5\lib;%PATH%"
	 set "QT_QPA_PLATFORM_PLUGIN_PATH=%EXTERNLIBS%\qt5\plugins\platforms"   rem tested for qt5 on win7, visual studio 2010
   ) ELSE (
     REM QTDIR is set so try to use it !
     REM Do a simple sanity-check...
     IF NOT EXIST "%QTDIR%\.qmake.cache" (
       echo *** WARNING: .qmake.cache NOT found !
       echo ***          Check QTDIR or simply do NOT set QT_HOME and QTDIR to use the version from EXTERNLIBS!
       pause
     )
     REM Set QT_HOME according to QTDIR. If User ignores any warnings before he will find himself in a world of pain! 
     set "QT_HOME=%QTDIR%"
     set "QT_SHAREDHOME=%QTDIR%"
     set "QT_INCPATH=%QTDIR%\include"
     set "QT_LIBPATH=%QTDIR%\lib"
	 set "PATH=%QTDIR%\bin;%QTDIR%\lib;%PATH%"
	 set "QT_QPA_PLATFORM_PLUGIN_PATH=%QTDIR%\plugins\platforms"  
   )
)

if not defined  OPENSCENEGRAPH_HOME (
REM   if exist %EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe (
REM     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --version-number') do @set OSG_VER_NUM=%%v
REM     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --so-number') do @set OSG_SO_VER=%%v
REM     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --openthreads-soversion-number') do @set OSG_OT_SO_VER=%%v
REM   )

   set "OPENSCENEGRAPH_HOME=%EXTERNLIBS%\OpenSceneGraph"
   set "OSG_DIR=%EXTERNLIBS%\OpenSceneGraph"
   set "OPENSCENEGRAPH_INCPATH=%EXTERNLIBS%\OpenSceneGraph\include"
   if "%USE_OPT_LIBS%" == "1" (
      set "OPENSCENEGRAPH_LIBS=%OSGNV_LIBS% -L%EXTERNLIBS%\OpenSceneGraph\lib -losg -losgDB -losgUtil -losgViewer -losgParticle -losgText -losgSim -losgGA -losgFX -lOpenThreads"
   ) else (
      set "OPENSCENEGRAPH_LIBS=%OSGNV_LIBS% -L%EXTERNLIBS%\OpenSceneGraph\lib -losgD -losgDBd -losgUtilD -losgViewerD -losgParticleD -losgTextD -losgSimD -losgGAd -losgFXd -lOpenThreadsD"
   )
)

if not defined ALVAR_HOME  (
   set "ALVAR_HOME=%EXTERNLIBS%\ALVAR"
   set "ALVAR_DEFINES=HAVE_ALVAR"
   set "ALVAR_INCPATH=%EXTERNLIBS%\ALVAR\include"
   set "ALVAR_PLUGIN_PATH=%COVISEDIR%\%ARCHSUFFIX%\lib\alvarplugins"
   set "ALVAR_LIBRARY_PATH=%EXTERNLIBS%\ALVAR\bin"
   
   
   set "PATHADD=%PATHADD%;%EXTERNLIBS%\ALVAR\bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "ALVAR_LIBS=-L%EXTERNLIBS%\ALVAR\lib -lALVARCollision -lALVARDynamics -lGIMPACTUtils -lLinearMath"
      set "OSGALVAR_LIBS=-L%EXTERNLIBS%\OpenSceneGraph\lib -losgbALVAR"
   ) else (
      set "ALVAR_LIBS=-L%EXTERNLIBS%\ALVAR\lib -lALVARCollisiond -lALVARDynamicsd -lGIMPACTUtilsd -lLinearMathd"
      set "OSGALVAR_LIBS=-L%EXTERNLIBS%\OpenSceneGraph\lib -losgbALVARd"
   )
)

if not defined PYTHONHOME  (
   set "PYTHONHOME=%EXTERNLIBS%\..\shared\Python;%EXTERNLIBS%\Python"
   rem PYTHON_HOME is for compiling Python 
   rem  while PYTHONHOME is for executing Python and can consist of
   rem several different paths
   rem set "PYTHONPATH=%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\Python"
   set "PYTHONPATH=%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\Python;%COVISEDIR%\PYTHON\bin;%COVISEDIR%\PYTHON\bin\vr-prepare;%COVISEDIR%\PYTHON\bin\vr-prepare\converters;%COVISEDIR%\PYTHON\bin\vr-prepare\negotiator;%COVISEDIR%\PYTHON\bin\vr-prepare\negotiator\import;%EXTERNLIBS%\pyqt\modules;%EXTERNLIBS%\sip\modules"
   set "PATH=%PATH%;%EXTERNLIBS%\Python\DLLs;%EXTERNLIBS%\Python;%EXTERNLIBS%\Python\bin"
)

if not defined ALL_EXTLIBS ( 
  set "ALL_EXTLIBS=%EXTERNLIBS%\all"
  set "PATH=%COVISEDIR%\%ARCHSUFFIX%\lib;%PATH%"
)

set FRAMEWORK=covise
set QMAKECOVISEDIR=%COVISEDIR%
set LOGNAME=covise
set PATH=%PATH%;%COVISEDIR%\%ARCHSUFFIX%\bin;%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\bin;%COVISEDIR%\%ARCHSUFFIX%\bin\Renderer;%COVISEDIR%\%ARCHSUFFIX%\lib\opencover\plugins


if not defined COVISEDESTDIR   set COVISEDESTDIR=%COVISEDIR%
if not defined VV_SHADER_PATH  set VV_SHADER_PATH=%COVISEDIR%\src\3rdparty\deskvox\virvo\shader
if not defined COVISE_PATH     set COVISE_PATH=%COVISEDIR%

set COMMON_ACTIVE=1
:END
