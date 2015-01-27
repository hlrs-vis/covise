@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM Sets environment variables for the trunk 
REM version of WIN32 COVISE
REM *******************************************

REM eval script arguments
IF "x%1x" EQU "xx" GOTO USAGE

REM get the 8.3 filename of current installation
SET TMPFILE="%TMP%\~covtmpfile.txt"
CALL "%WINDIR%\system32\cscript.exe" /nologo "%~dp0\get8dot3path.vbs" "%~dp0">%TMPFILE%
SET /P INSTDIR=<%TMPFILE%
DEL /Q %TMPFILE%

IF "x%2x" EQU "xx" (
   SET COVISEDIR=%INSTDIR%
) ELSE ( SET "COVISEDIR=%2" )

SET ARCHSUFFIX=%1
SET __QUIETMODE=FALSE
IF "x%3x" EQU "xx" GOTO SKIP_SETQUIETMODE
   IF "x%3x" NEQ "x/Qx" GOTO USAGE
   SET __QUIETMODE=TRUE
   SET __QUIETPARM=--quiet
   GOTO SKIP_MESSAGE1
:SKIP_SETQUIETMODE
   ECHO Executing common.VISENSO.bat in directory %~dp0
:SKIP_MESSAGE1


REM reset variables possibly existing from previous calls
REM todo: warn, if any of these variables already exist, since PATH could
REM be "polluted" by previous COVISE runs
SET INNOSETUPHOME=
SET PYTHONHOME=
SET PYTHONVERSION=
SET QTDIR=
SET COVISE_BRANCH=
SET COVISE_PATH=
SET COFRAMEWORKDIR=
SET COVISECONFIG_DEBUG=
SET QMAKESPEC=
SET DXSDK_DIR=
SET DSV_HOME=




REM if such a file is found, execute it to set the custom application
REM paths on this system 
IF EXIST %INSTDIR%\common.local.bat (
   IF "%__QUIETMODE%" EQU "FALSE" (
      ECHO ...setting application paths by file %INSTDIR%\common.local.bat...
   )
   CALL %INSTDIR%\common.local.bat
) ELSE (
   IF "%__QUIETMODE%" EQU "FALSE" (
      ECHO ...setting internal application paths ...
   )
)


IF "x%EXTERNLIBS%x" EQU "xx" GOTO NOEXTERNLIBSSET
IF "x%COVISE_BRANCH%x" EQU "xx" SET COVISE_BRANCH=VISENSO
IF "x%COVISE_PATH%x" EQU "xx" SET "COVISE_PATH=%COVISEDIR%"
IF "x%COFRAMEWORKDIR%x" EQU "xx" SET "COFRAMEWORKDIR=%COVISEDIR%"
IF "x%COVISECONFIG_DEBUG%x" EQU "xx" SET COVISECONFIG_DEBUG=1


IF "x%PYTHONHOME%x" EQU "xx" SET "PYTHONHOME=%EXTERNLIBS%\Python"
SET "PYTHON_HOME=%PYTHONHOME%"
REM PYTHON_HOME is for compiling Python while PYTHONHOME is for 
REM executing Python and can consist of several different paths
SET "PYTHONPATH=%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\Python;%PYTHONHOME%\DLLs;%PYTHONHOME%\Lib"
SET "PYTHON_INCLUDE=%PYTHONHOME%\Include"
REM SET "PYTHON_INCLUDE=%PYTHONHOME%\Include %PYTHONHOME%\PC"
IF "x%PYTHONVERSION%x" EQU "xx" SET PYTHONVERSION=25
IF "x%ARCHSUFFIX:~-3,3%x" NEQ "xoptx" SET PYTHONVERSION=%PYTHONVERSION%_d
SET "PYTHON_LIB=%PYTHONHOME%\DLLs\python%PYTHONVERSION%.lib"
SET "COVISE_LOCAL_PYTHON=%PYTHONHOME%\DLLs\python.exe"
SET "PYTHON_LIBS=%PYTHONHOME%\DLLs\python%PYTHONVERSION%.lib"
SET "PATH=%PYTHONHOME%;%PYTHONHOME%\DLLs;%COVISEDIR%\Python;%PATH%"
SET "PYTHON_INCPATH=%PYTHON_HOME%\include"
REM SET "PYTHON_LIBPATH=%PYTHONHOME%\DLLs"


IF "x%QMAKESPEC%x" EQU "xx" SET QMAKESPEC=win32-msvc2005
IF "x%QTDIR%x" EQU "xx" SET "QTDIR=%EXTERNLIBS%\Qt"


IF "%__QUIETMODE%" EQU "FALSE" (
   REM Do a simple sanity-check...
   IF NOT EXIST "%QTDIR%\.qmake.cache" (
      ECHO  *** WARNING: .qmake.cache NOT found !
      ECHO  ***          Check QTDIR or simply do NOT set QT_HOME and QTDIR to use the version from EXTERNLIBS!
      REM PAUSE
   )
)
REM Set QT_HOME according to QTDIR. If User ignores any warnings before he will find himself in a world of pain! 
SET "QT_HOME=%QTDIR%"
SET "QT_SHAREDHOME=%QTDIR%"
SET "QT_INCPATH=%QTDIR%\include"
SET "QT_LIBPATH=%QTDIR%\lib"
SET "PATH=%QTDIR%\bin;%QTDIR%\lib;%PATH%"


IF "x%DXSDK_DIR%x" EQU "xx" SET "DXSDK_DIR=%EXTERNLIBS%\DXSDK\"
SET "DXSDK_HOME=%DXSDK_DIR%" 
SET "DXSDK_INCPATH=%DXSDK_DIR%include"
SET "DXSDK_LIBS=-L%DXSDK_DIR%lib -L%DXSDK_DIR%lib\x86 -ldxguid -ldxerr9 -ldinput8 -lcomctl32"


REM SET "SWIG=%EXTERNLIBS%\swig\swig.exe"
REM SET "SWIG_HOME=%EXTERNLIBS%\SWIG"

REM SET TIFF_DEFINES=HAVE_LIBTIF
REM SET "TIFF_HOME=%EXTERNLIBS%\tiff"
REM SET "TIFF_INCPATH=%EXTERNLIBS%\tiff\include"
REM SET "TIFF_LIBPATH=%EXTERNLIBS%\tiff\lib"
REM SET TIFF_LIBRARIES=-llibtiff
REM SET "TIFF_LIBS=-L%TIFF_LIBPATH% -llibtiff"

REM SET OSG_PATH=
REM SET "PNG_LIBS=%EXTERNLIBS%\png\lib"
REM SET "PRODUCER_LIBPATH=%EXTERNLIBS%\Producer\lib\win32"
REM SET "ZLIB_LIBS=%EXTERNLIBS%\zlib\lib"
IF "x%DSV_HOME%x" EQU "xx" SET "DSV_HOME=%EXTERNLIBS%\DSVL"

SET "PATH=%PATH%;%INNOSETUPHOME%"
REM SET "PATH=%QTDIR%\bin;%PATH%"
REM SET "PATH=%EXTERNLIBS%\xerces\bin;%PATH%"
REM SET "PATH=%EXTERNLIBS%\UnixUtils\usr\local\wbin;%EXTERNLIBS%\UnixUtils\bin;%PATH%"
SET "PATH=%EXTERNLIBS%\openssl\bin;%PATH%"
REM SET "PATH=%PYTHONHOME%\PCBuild8\win32release;%PATH%"
REM SET "PATH=%PYTHONHOME%;%PATH%"
REM SET "PATH=%OSG_PATH%;%PATH%"
REM SET "PATH=%COVISEDIR%\src\visenso\NS3\java_interfaces\Release;%PATH%"
SET "PATH=%EXTERNLIBS%\libxml2\lib;%PATH%"
SET "PATH=%EXTERNLIBS%\glew\lib;%PATH%"

REM common.bat also calls %COVISEDIR%\bin\common-base.bat
CALL %COVISEDIR%\common.bat %ARCHSUFFIX% %__QUIETPARM%

IF "%__QUIETMODE%" EQU "TRUE" GOTO SKIP_VARIABLEECHO
   ECHO ARCHSUFFIX=%ARCHSUFFIX%
   ECHO COVISEDIR=%COVISEDIR%
   ECHO EXTERNLIBS=%EXTERNLIBS%
   ECHO COVISE_BRANCH=%COVISE_BRANCH%
   ECHO COVISE_PATH=%COVISE_PATH%
   ECHO COFRAMEWORKDIR=%COFRAMEWORKDIR%
   ECHO COVISECONFIG_DEBUG=%COVISECONFIG_DEBUG%
   REM ECHO PATH=%PATH%
   ECHO PYTHONHOME=%PYTHONHOME%
   ECHO PYTHON_HOME=%PYTHON_HOME%
   ECHO PYTHONPATH=%PYTHONPATH%
   ECHO PYTHON_INCLUDE=%PYTHON_INCLUDE%
   ECHO PYTHON_LIB=%PYTHON_LIB%
   ECHO COVISE_LOCAL_PYTHON=%COVISE_LOCAL_PYTHON%
   ECHO PYTHON_LIBS=%PYTHON_LIBS%
   ECHO PYTHON_INCPATH=%PYTHON_INCPATH%
   ECHO PYTHON_LIBPATH=%PYTHON_LIBPATH%
   ECHO PYTHON_PATH=%PYTHON_PATH%
   ECHO QMAKESPEC=%QMAKESPEC%
   ECHO QTDIR=%QTDIR%
   ECHO QT_HOME=%QT_HOME%
   ECHO QT_SHAREDHOME=%QT_SHAREDHOME%
   ECHO QT_INCPATH=%QT_INCPATH%
   ECHO QT_LIBPATH=%QT_LIBPATH%
   ECHO DXSDK_DIR=%DXSDK_DIR%
   ECHO DXSDK_HOME=%DXSDK_HOME% 
   ECHO DXSDK_INCPATH=%DXSDK_INCPATH%
   ECHO DXSDK_LIBS=%DXSDK_LIBS%
   ECHO SWIG=%SWIG%
   ECHO SWIG_HOME=%SWIG_HOME%
   ECHO TIFF_DEFINES=%TIFF_DEFINES%
   ECHO TIFF_HOME=%TIFF_HOME%
   ECHO TIFF_INCPATH=%TIFF_INCPATH%
   ECHO TIFF_LIBPATH=%TIFF_LIBPATH%
   ECHO TIFF_LIBRARIES=%TIFF_LIBRARIES%
   ECHO TIFF_LIBS=%TIFF_LIBS%
   ECHO OSG_PATH=%OSG_PATH%
   ECHO PNG_LIBS=%PNG_LIBS%
   ECHO PRODUCER_LIBPATH=%PRODUCER_LIBPATH%
   ECHO ZLIB_LIBS=%ZLIB_LIBS%
   ECHO DSV_HOME=%DSV_HOME%
   ECHO GLEW_HOME=%GLEW_HOME%
   ECHO INNOSETUPHOME=%INNOSETUPHOME%
   ECHO OPENSCENEGRAPH_HOME=%OPENSCENEGRAPH_HOME%
    ECHO OPENSCENEGRAPH_lIBS=%OPENSCENEGRAPH_LIBS%
:SKIP_VARIABLEECHO

GOTO END



:NOEXTERNLIBSSET
ECHO ...
ECHO ERROR: no EXTERNLIBS is set! Set systemvariable EXTERNLIBS to the directory
ECHO    where the external COVISE libraries are installed.
ECHO ...
GOTO USAGE



:USAGE
ECHO ...
ECHO Sets environment variables for the trunk 
ECHO version of WIN32 COVISE
ECHO ...
ECHO usage:
ECHO common.VISENSO.bat 
ECHO    [ARCHSUFFIX] -- COVISE architecture suffix
ECHO    [COVISEDIR] -- install path of covise sources (e. g. D:\TRUNK\covise)
ECHO    [/Q] -- optional: quiet mode; do not ECHO anything
ECHO ...
ECHO note: environment variable EXTERNLIBS is expected to be 
ECHO    set to the path of external libraries COVISE depends upon
ECHO ...
ECHO called batch scripts:
ECHO     %COVISEDIR%\common.bat
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
EXIT /B 1


:END
EXIT /B 0

