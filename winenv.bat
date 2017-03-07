@echo off

if defined COMMON_ACTIVE (
   goto END
)

if "%1" == "--help" (
   echo common.bat [ARCHSUFFIX]
   echo "ARCHSUFFIX: win32, win32opt, amdwin64, amdwin64opt, ia64win, vista (default), vistaopt, zackel, zackelopt, angus, angusopt, yoroo, yorooopt, berrenda, berrendaopt, tamarau, tamarauopt,zebu, zebuopt mingw, mingwopt"
   pause
   goto END
)

if "%1" == "" (
  if not defined ARCHSUFFIX% set ARCHSUFFIX=zebuopt
) else (
  set ARCHSUFFIX=%1
)

set BASEARCHSUFFIX=%ARCHSUFFIX:opt=%

if not defined EXTERNLIBS (
   if not defined EXTERNLIBSROOT (
      echo EXTERNLIBS and EXTERNLIBSROOT are not set
      pause
      goto END
   ) else (
      set EXTERNLIBS=%EXTERNLIBSROOT%\%BASEARCHSUFFIX%
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
   cd /d %CADCV_DIR%\qmakehelp
   call qmakehelp.bat
   cd /d %COVISEDIR%
)

if not defined COFRAMEWORKDIR (
   set COFRAMEWORKDIR=%COVISEDIR%
)

if /I "%BASEARCHSUFFIX%" EQU "amdwin64" goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "ia64win"  goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "win32"    goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "vista"    goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "zackel"    goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "angus"    goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "yoroo"    goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "berrenda"    goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "tamarau"    goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "zebu"    goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "mingw"    goto LABEL_SETENVIRONMENT
echo ARCHSUFFIX %ARCHSUFFIX% is not supported!
echo common.bat [ARCHSUFFIX]
echo "ARCHSUFFIX: win32, win32opt, amdwin64, amdwin64opt, ia64win, vista (default), vistaopt, zackel, zackelopt, angus, angusopt, yoroo, yorooopt, berrenda, berrendaopt, tamarau, tamarauopt, zebu, zebuopt, mingw, mingwopt"
pause
goto END



:LABEL_SETENVIRONMENT

echo Environment settings for ARCHSUFFIX %ARCHSUFFIX%

cd /d %COVISEDIR%
         
if exist "%COVISEDIR%"\mysetenv.bat (
   call "%COVISEDIR%"\mysetenv.bat
   echo mysetenv.bat was included
)


rem   start microsoft development environment
rem   =======================================
rem
rem If VS2003 or VS2005 was installed in a non-standard location you have to set VCVARS32 !
rem 

set PROGFILES=%ProgramFiles%
if defined ProgramFiles(x86)  set PROGFILES=%ProgramFiles(x86)%
rem echo  %VS100COMNTOOLS%
cd

if "%BASEARCHSUFFIX%" EQU "win32" (
    call "%PROGFILES%"\"Microsoft Visual Studio .NET 2003"\Vc7\bin\vcvars32.bat
) else if "%BASEARCHSUFFIX%" EQU "vista" (
    call "%VS80COMNTOOLS%"\..\..\Vc\bin\vcvars32.bat"
) else if "%BASEARCHSUFFIX%" EQU "zackel" (
    cd /d "%VS90COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x86
    cd /d "%COVISEDIR%"\
) else if "%BASEARCHSUFFIX%" EQU "yoroo" (
    cd /d "%VS100COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x86
    cd /d "%COVISEDIR%"\
) else if "%BASEARCHSUFFIX%" EQU "tamarau" (
    cd /d "%VS110COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x64
    cd /d "%COVISEDIR%"\
) else if "%BASEARCHSUFFIX%" EQU "zebu" (
    cd /d "%VS140COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x64
    cd /d "%COVISEDIR%"\
) else if "%BASEARCHSUFFIX%" EQU "berrenda" (
if defined VS110COMNTOOLS  (
    cd /d "%VS110COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x64
) else (
    cd /d "%VS100COMNTOOLS%"\..\..\vc
	call vcvarsall.bat x64
	)
    cd /d "%COVISEDIR%"\
) else if "%BASEARCHSUFFIX%" EQU "angus" (
    cd /d "%VS90COMNTOOLS%"\..\..\vc
    call vcvarsall.bat x64
    cd /d "%COVISEDIR%"\
) else if "%BASEARCHSUFFIX%" EQU "amdwin64"   (
    cd /d "%VS80COMNTOOLS%"\..\..\vc
    call vcvarsall.bat x64
    cd /d "%COVISEDIR%"\
) else if defined VCVARS32 (
    call "%VCVARS32%"
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
   if exist %EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe (
     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --version-number') do @set OSG_VER_NUM=%%v
     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --so-number') do @set OSG_SO_VER=%%v
     for /f %%v in ('%EXTERNLIBS%\OpenSceneGraph\bin\osgversion.exe --openthreads-soversion-number') do @set OSG_OT_SO_VER=%%v
   )

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
   set "ALVAR_PLUGIN_PATH=%EXTERNLIBS%\ALVAR\bin\alvarplugins"
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

if not defined CUDA_HOME  (
   if defined CUDA_BIN_PATHDONTBECAUSEOFSPACES  (
      REM set "CUDA_HOME=%CUDA_INC_PATH%\.."
      REM set "CUDA_INCPATH=%CUDA_INC_PATH%"
      set "CUDA_HOME=%EXTERNLIBS%\Cuda"
      set "CUDA_BIN_PATH=%EXTERNLIBS%\Cuda\bin"
      set "CUDA_INCPATH=%EXTERNLIBS%\Cuda\include"
      set "CUDA_DEFINES=HAVE_CUDA"
      set "CUDA_SDK_HOME=%EXTERNLIBS%\CUDA"
      set "CUDA_SDK_INCPATH=%EXTERNLIBS%\CUDA\include %EXTERNLIBS%\CUDA\common\inc"
      set "PATHADD=%PATHADD%;%CUDA_BIN_PATH%"
      if "%USE_OPT_LIBS%" == "1" (
	     set "PATHADD=%PATHADD%;%EXTERNLIBS%\Cuda\bin\win32\Release;%EXTERNLIBS%\cudpp\bin"
         set "CUDA_LIBS=-L%CUDA_LIB_PATH% -L%EXTERNLIBS%\Cuda\common\lib -lcuda -lcudart -lcutil32"
      ) else (
	     set "PATHADD=%PATHADD%;%EXTERNLIBS%\Cuda\bin\win32\Debug;%EXTERNLIBS%\cudpp\bin"
         set "CUDA_LIBS=-L%CUDA_LIB_PATH% -L%EXTERNLIBS%\Cuda\common\lib -lcuda -lcudart -lcutil32D"
      )
   ) else if exist %EXTERNLIBS%\Cuda (
      set "CUDA_HOME=%EXTERNLIBS%\Cuda"
      set "CUDA_BIN_PATH=%EXTERNLIBS%\Cuda\bin"
      set "CUDA_INCPATH=%EXTERNLIBS%\Cuda\include"
      set "CUDA_DEFINES=HAVE_CUDA"
      set "CUDA_SDK_HOME=%EXTERNLIBS%\CUDA"
      set "CUDA_SDK_INCPATH=%EXTERNLIBS%\CUDA\include %EXTERNLIBS%\CUDA\common\inc"
      set "PATHADD=%PATHADD%;%CUDA_BIN_PATH%"
      if "%USE_OPT_LIBS%" == "1" (
	     set "PATHADD=%PATHADD%;%EXTERNLIBS%\Cuda\bin\win32\Release"
         set "CUDA_LIBS=-L%EXTERNLIBS%\Cuda\lib -L%EXTERNLIBS%\Cuda\common\lib -lcuda -lcudart -lcutil64"
      ) else (
	     set "PATHADD=%PATHADD%;%EXTERNLIBS%\CUDA\bin\win32\Debug"
         set "CUDA_LIBS=-L%EXTERNLIBS%\Cuda\lib -L%EXTERNLIBS%\Cuda\common\lib -lcuda -lcudart -lcutil64D"
      )
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
  set "PATH=%EXTERNLIBS%\all;%PATH%"
)

if not defined HDF5_ROOT  (
   set "HDF5_ROOT=%EXTERNLIBS%\hdf5"
)
if not defined Qt5WebEngineWidgets_DIR  (
   set "Qt5WebEngineWidgets_DIR=%EXTERNLIBS%\qt5"
)


set LOGNAME=covise
set PATH=%PATH%;%COVISEDIR%\%ARCHSUFFIX%\bin;%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\bin;%COVISEDIR%\%ARCHSUFFIX%\bin\Renderer;%COVISEDIR%\%ARCHSUFFIX%\lib\opencover\plugins
set PATH=%PATH%;%EXTERNLIBS%\bison\bin

if not defined COVISEDESTDIR   set COVISEDESTDIR=%COVISEDIR%
if not defined VV_SHADER_PATH  set VV_SHADER_PATH=%COVISEDIR%\src\3rdparty\deskvox\virvo\shader
if not defined COVISE_PATH     set COVISE_PATH=%COVISEDESTDIR%;%COVISEDIR%


set RM=rmdir /S /Q
if "%ARCHSUFFIX%" EQU "vista"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "zackel"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "angus"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "yoroo"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "berrenda"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "tamarau"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "zebu"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "mingw"  set COVISE_DEVELOPMENT=YES

set COMMON_ACTIVE=1
:END
