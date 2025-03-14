@echo off

if defined COMMON_ACTIVE (
   goto END
)

set VCPKG_DEFAULT_TRIPLET=x64-windows
set VCPKG_OSGVER=3.6.4
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
set COVISE_ARCHSUFFIX=%ARCHSUFFIX%

set BASEARCHSUFFIX=%ARCHSUFFIX:opt=%

if not defined EXTERNLIBS (
   if not defined EXTERNLIBSROOT (
      echo EXTERNLIBS and EXTERNLIBSROOT are not set
      if "%BASEARCHSUFFIX%" NEQ "vcpkg" (
         pause
         goto END
      )
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
if /I "%BASEARCHSUFFIX%" EQU "uwp"    goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "mingw"    goto LABEL_SETENVIRONMENT
if /I "%BASEARCHSUFFIX%" EQU "vcpkg"    goto LABEL_SETENVIRONMENT
echo ARCHSUFFIX %ARCHSUFFIX% is not supported!
echo common.bat [ARCHSUFFIX]
echo "ARCHSUFFIX: vcpkg, vcpkgopt, win32, win32opt, amdwin64, amdwin64opt, ia64win, vista (default), vistaopt, zackel, zackelopt, angus, angusopt, yoroo, yorooopt, berrenda, berrendaopt, tamarau, tamarauopt, zebu, zebuopt, mingw, mingwopt"
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

if "%BASEARCHSUFFIX%" EQU "zebu"  (
    set VC14_15=yes
) else if "%BASEARCHSUFFIX%" EQU "uwp"  (
    set VC14_15=yes
) else if "%BASEARCHSUFFIX%" EQU "vcpkg"  (
    set VC14_15=yes
)

if "%VC14_15%" EQU "yes" (
   if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" ( 
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
   ) else if exist "D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" ( 
    call "D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
   ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" ( 
	call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 
   ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" ( 
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
   ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsDevCmd.bat" ( 
    call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 
   ) 
       
    
    if NOT defined VS150COMNTOOLS% (
      if NOT defined VS160COMNTOOLS% (
        if NOT defined VS170COMNTOOLS% ( 
            cd /d "%VS140COMNTOOLS%"\..\..\vc
	        call vcvarsall.bat x64
            cd /d "%COVISEDIR%"\
	    )
	)
	)
) else if "%BASEARCHSUFFIX%" EQU "win32" (
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
if "%BASEARCHSUFFIX%" EQU "vcpkg" (
    if "%VCPKG_ROOT%" EQU "" (
        set "VCPKG_ROOT=C:\vcpkg"
    )
)

if "%BASEARCHSUFFIX%" EQU "vcpkg" (
    if NOT EXIST "%VCPKG_ROOT%\.vcpkg-root" (
        VCPKG_ROOT has to be set to the root directory of your vcpkg installation
        set VCPKG_ROOT=
    )
)

if "%VCPKG_ROOT%" NEQ "" (
    if "%BASEARCHSUFFIX%" EQU "vcpkg" (
        set "PATH=%VCPKG_ROOT%\installed\%VCPKG_DEFAULT_TRIPLET%\bin;%VCPKG_ROOT%;%PATH%"
        set "OSG_LIBRARY_PATH=%VCPKG_ROOT%\installed\%VCPKG_DEFAULT_TRIPLET%\tools\osg\osgPlugins-%VCPKG_OSGVER%"
    )
)

if "%VCPKG_ROOT%" NEQ "" (
    if "%ARCHSUFFIX%" EQU "vcpkg" (
          set "PATH=%VCPKG_ROOT%\installed\x64-windows\debug\bin;%PATH%"
          set "OSG_LIBRARY_PATH=%VCPKG_ROOT%\installed\%VCPKG_DEFAULT_TRIPLET%\debug\tools\osg\osgPlugins-%VCPKG_OSGVER%"
    )
)
set "CMAKE_CONFIGURATION_TYPES=Debug;Release"

if "%BASEARCHSUFFIX%" EQU "vcpkg" (
    goto FINALIZE
)

if defined CUDA_PATH_V10_0 (
    set CUDA_PATH=%CUDA_PATH_V10_0%
)
if defined CUDA_PATH_V10_1 (
    set CUDA_PATH=%CUDA_PATH_V10_1%
)
if not defined QT_HOME ( 
   REM QT_HOME is not set... check QTDIR
   IF not defined QTDIR (
     REM QTDIR is not set ! Try in EXTERNLIBS
     IF "%2"=="qt6" (
        set "QTDIR=%EXTERNLIBS%\qt6"
        set "COVISE_CMAKE_OPTIONS=%COVISE_CMAKE_OPTIONS% -DCOVISE_USE_QT5=OFF"
     ) ELSE (
        set "QTDIR=%EXTERNLIBS%\qt5"
     )
   ) ELSE (
     REM QTDIR is set so try to use it !
     REM Do a simple sanity-check...
     IF NOT EXIST "%QTDIR%\.qmake.cache" (
       echo *** WARNING: .qmake.cache NOT found !
       echo ***          Check QTDIR or simply do NOT set QT_HOME and QTDIR to use the version from EXTERNLIBS!
       pause
     )
     REM Set QT_HOME according to QTDIR. If User ignores any warnings before he will find himself in a world of pain! 
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

if exist %EXTERNLIBS%\cef\Release\libcef.dll (
   if "%USE_OPT_LIBS%" == "1" (
	 set "PATH=%PATH%;%EXTERNLIBS%\cef\Release"
   ) else (
	 set "PATH=%PATH%;%EXTERNLIBS%\cef\Debug"
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
   set "PYTHONHOME=%EXTERNLIBS%\Python"
   rem PYTHON_HOME is for compiling Python 
   rem  while PYTHONHOME is for executing Python and can consist of
   rem several different paths
   rem set "PYTHONPATH=%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\Python"
   set "PYTHONPATH=%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\Python;%COVISEDIR%\PYTHON\bin;%COVISEDIR%\PYTHON\bin\vr-prepare;%COVISEDIR%\PYTHON\bin\vr-prepare\converters;%COVISEDIR%\PYTHON\bin\vr-prepare\negotiator;%COVISEDIR%\PYTHON\bin\vr-prepare\negotiator\import;%EXTERNLIBS%\pyqt\modules;%EXTERNLIBS%\sip\modules"
   set "PATH=%EXTERNLIBS%\Python\bin;%EXTERNLIBS%\Python\DLLs;%EXTERNLIBS%\Python;%PATH%"
)


if "%USE_OPT_LIBS%" == "1" (
  set "PATHADD=%PATHADD%;%EXTERNLIBS%\PhysX\bin\win.x86_64.vc142.md\release"
) else (
  set "PATHADD=%PATHADD%;%EXTERNLIBS%\PhysX\bin\win.x86_64.vc142.md\debug"
)
if not defined ALL_EXTLIBS ( 
  set "ALL_EXTLIBS=%EXTERNLIBS%\all"
  set "PATH=%EXTERNLIBS%\all;%PATH%"
)

if not defined HDF5_ROOT  (
   set "HDF5_ROOT=%EXTERNLIBS%\hdf5"
)
if not "%COVISE_USE_QT5%" == "OFF" if not defined Qt5WebEngineWidgets_DIR  (
   set "Qt5WebEngineWidgets_DIR=%QTDIR%"
)

set PATH=%PATH%;%EXTERNLIBS%\bison\bin

set "QT_HOME=%QTDIR%"
set "QT_SHAREDHOME=%QT_HOME%"
set "QT_INCPATH=%QT_HOME%\include"
set "QT_LIBPATH=%QT_HOME%\lib"
set "QT_PLUGIN_PATH=%QT_HOME%\plugins"
set "QT_RESOURCES_PATH=%QT_HOME%\resources"
set "QTDIR=%QT_HOME%"
set "PATH=%QT_HOME%\bin;%QT_HOME%\lib;%PATH%"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%QT_HOME%\plugins\platforms"  
set "QTWEBENGINE_RESOURCES_PATH=%QT_HOME%\resources"
rem set "QTWEBENGINEPROCESS_PATH=%QT_HOME%\bin\QtWebEngineProcess.exe"
set QTWEBENGINE_DISABLE_SANDBOX=1

:FINALIZE
set LOGNAME=covise
set PATH=%PATH%;%COVISEDIR%\%ARCHSUFFIX%\bin;%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\bin;%COVISEDIR%\%ARCHSUFFIX%\bin\Renderer;%COVISEDIR%\%ARCHSUFFIX%\lib\opencover\plugins;C:\Program Files\NVIDIA Corporation\NVSMI

if not defined COVISEDESTDIR   set COVISEDESTDIR=%COVISEDIR%
if not defined COVISE_PATH (
   if "%COVISEDESTDIR%" EQU "%COVISEDIR%" (
       set "COVISE_PATH=%COVISEDIR%"
   ) else (
       set COVISE_PATH=%COVISEDESTDIR%;%COVISEDIR%
   )
)


set RM=rmdir /S /Q
if "%ARCHSUFFIX%" EQU "vista"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "zackel"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "angus"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "yoroo"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "berrenda"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "tamarau"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "zebu"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "vcpkg"  set COVISE_DEVELOPMENT=YES
if "%ARCHSUFFIX%" EQU "mingw"  set COVISE_DEVELOPMENT=YES

set COMMON_ACTIVE=1
:END

cd /d %COVISEDIR%
