@ECHO OFF

REM ************************************************************
REM 
REM This script starts a specified COVISE python script.
REM
REM Usage: 
REM    runCOVISEScriptIF 
REM       <name of python script>  
REM       <script argument 1> 
REM       ... 
REM       <script argument n>
REM
REM (C) Copyright 2009 VISENSO GmbH
REM
REM ************************************************************

REM changes to environment variables are only local
SETLOCAL ENABLEDELAYEDEXPANSION

IF [%1]==[] GOTO USAGE

ECHO Starting COVISE python script %1 %2 %3 %4 %5 %6 %7 %8 %9
ECHO ... COVISEDIR=%COVISEDIR%
ECHO ... ARCHSUFFIX=%ARCHSUFFIX%
ECHO ... EXTERNLIBS=%EXTERNLIBS%
ECHO ... PYTHONPATH=%PYTHONPATH%
ECHO ... PYTHONHOME=%PYTHONHOME%
ECHO ... LOCAL_COVISE_HOME=%LOCAL_COVISE_HOME%
ECHO ... COCONFIG=%COCONFIG%

SET "COVISE_PATH=%COVISEDIR%;%LOCAL_COVISE_HOME%"
SET "PATH=%COVISEDIR%;%PATH%"
SET "PATH=%LOCAL_COVISE_HOME%;%PATH%"

REM Todo: iterate over all paths included in COVISE_PATH
SET "libPathName=%EXTERNLIBS%\xerces\lib;%EXTERNLIBS%\QT-4.4.3\lib;%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\%ARCHSUFFIX"

REM adapt Python environment for COVISE
IF "x%ARCHSUFFIX:~-3,3%x" EQU "xoptx" SET "_PYTHON=%PYTHONHOME%\python.exe"
IF "x%ARCHSUFFIX:~-3,3%x" NEQ "xoptx" SET "_PYTHON=%PYTHONHOME%\python_d.exe"
IF DEFINED COVISE_LOCAL_PYTHON        SET "_PYTHON=%COVISE_LOCAL_PYTHON%"
ECHO ... _PYTHON=%_PYTHON%
SET "PYTHONPATH=%COVISEDIR%\Python;%COVISEDIR%/%ARCHSUFFIX%/bin;%COVISEDIR%/%ARCHSUFFIX%/lib"

SET "_UNIXSHELL=%EXTERNLIBS%\UnixUtils\sh.exe"
ECHO ... _UNIXSHELL=%_UNIXSHELL%

SET "CFX5_UNITS_DIR=%EXTERNLIBS%\CFX-5"
SET "COVISE_CATIA_SERVER=obie"
SET COVISE_CATIA_SERVER_PORT=7000
SET "INX_CADADAPTER_CONFIGDIR=%EXTERNLIBS%\..\shared\INCENTRIX"
SET INX_CADADAPTER_LOGCONFIGFILE=log4cpp.ini
SET INX_CADADAPTER_LOGDIR=/var/tmp/

REM Set this variable, if there is no VRML license, but the customer uses COVER
REM SET COVER_NO_VRML=1

REM Set this variable, if an NVidia GPU is available
SET __GL_FSAA_MODE=7

REM Path abbreviations
REM SET "_UIBASEDIR=%COVISEDIR%\src\visenso\ui"
REM SET "_UIQT4BASE=%COVISEDIR%\src\visenso\branches\pyqt4"
SET "_UIBASEDIR=%COVISEDIR%\Python\bin\vr-prepare"
SET "_UIQT4BASE=%_UIBASEDIR%"


REM Set =1 if argument 1 is to be treated as a binary executable.
SET _DEFAULTCASE=1


IF /I "%1" NEQ "vr-prepare" GOTO SKIP_VRPREPARE
   ECHO ...starting vr-prepare
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\ui\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\negotiator\import
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare
   SET PATH=%PATH%;%PYTHONPATH%
   REM Check if there are parameters and if the first parameter is an existing file
   IF NOT [%2]==[] (
      IF EXIST "%2" (
         SET VR_PREPARE_PROJECT=%2
      ) else (
         ECHO ERROR
         ECHO      File does not exist: %2
         ECHO ERROR
         SET VR_PREPARE_PROJECT=
      )
   )
   
   REM ECHO ...starting valgrind
   REM valgrind --trace-children=yes --log-file=test
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIBASEDIR%\vr-prepare\vr-prepare.py
   
   SET _DEFAULTCASE=0
:SKIP_VRPREPARE


IF "%1" NEQ "vr-prepare4" GOTO SKIP_VRPREPARE4
   ECHO ...starting vr-prepare4
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\ui\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%\negotiator\import
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%
   SET PATH=%PATH%;%PYTHONPATH%
   REM Check if there are parameters and if the first parameter is an existing file
   IF NOT [%2] == [] (
      IF EXIST "%2" (
         SET VR_PREPARE_PROJECT=%2
      ) else (
         ECHO ERROR
         ECHO      File does not exist: %2
         ECHO ERROR
         SET VR_PREPARE_PROJECT=
      )
   )
   
   REM ECHO ...starting valgrind
   REM valgrind --trace-children=yes --log-file=test
   REM SET COVISE_LOCAL_PYTHON=
   REM %COVISEDIR%\%ARCHSUFFIX%/bin/covise --script %_UIBASEDIR%\vr-prepare\vr-prepare.py
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIQT4BASE%\vr-prepare.py
   REM %_UIQT4BASE%\MainWindow.py
   REM %_PYTHON% %_UIQT4BASE%\negotiator\import\ImportManager.py
   
   SET _DEFAULTCASE=0
:SKIP_VRPREPARE4


IF "%1" NEQ "reducer" GOTO SKIP_REDUCER
   
   ECHO ...starting reducer
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\converters
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare
   
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIBASEDIR%\vr-prepare\converters\reducer.py
   
   SET _DEFAULTCASE=0
:SKIP_REDUCER


IF "%1" NEQ "celltovert" GOTO SKIP_CELLTOVERT
   
   ECHO ...starting celltovert
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\converters
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare
   
   SET CONVERTFILES=`echo $@`
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIBASEDIR%\vr-prepare\converters\celltovert.py
   
   SET _DEFAULTCASE=0
:SKIP_CELLTOVERT


IF "%1" NEQ "scaleVector" GOTO SKIP_SCALEVECTOR
   
   ECHO ...starting scaleVector
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\converters
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare
   
   SET CONVERTFILES=`echo $@`
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIBASEDIR%\vr-prepare\converters\scaleVector.py
   
   SET _DEFAULTCASE=0
:SKIP_SCALEVECTOR


IF "%1" NEQ "make_transient" GOTO SKIP_MAKETRANSIENT
   
   ECHO ...starting make_transient
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\converters
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare
   
   SET CONVERTFILES=`echo $@`
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIBASEDIR%\vr-prepare\converters\make_transient.py
   
   SET _DEFAULTCASE=0
:SKIP_MAKETRANSIENT


IF "%1" NEQ "transform" GOTO SKIP_TRANSFORM
   
   ECHO ...starting transform
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\converters
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare
   
   SET CONVERTFILES=`echo $@`
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIBASEDIR%\vr-prepare\converters\transform.py
   
   SET _DEFAULTCASE=0
:SKIP_TRANSFORM


IF "%1" NEQ "calcCovise" GOTO SKIP_CALCCOVISE
   
   ECHO ...starting calcCovise
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare\converters
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare
   
   SET CONVERTFILES=`echo $@`
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIBASEDIR%\vr-prepare\converters\calcCovise.py
   
   SET _DEFAULTCASE=0
:SKIP_CALCCOVISE


IF "%1" NEQ "coCaseEditor" GOTO SKIP_COCASEEDITOR
   
   ECHO ...starting coCaseEditor
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\ui\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%\negotiator\import
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%
   SET PATH=%PATH%;%PYTHONPATH%
   
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIBASEDIR%\CocaseEditor.py
   
   SET _DEFAULTCASE=0
:SKIP_COCASEEDITOR


IF "%1" NEQ "cfx2covise.sh" GOTO SKIP_CFX2COVISESH
   
   ECHO ...starting cfx2covise.sh
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\vr-prepare\converters
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare
   
   %_UNIXSHELL% %_UIBASEDIR%\vr-prepare\converters\cfx2covise.sh %2 %3 %4 %5 %6 %7 %8 %9
   
   SET _DEFAULTCASE=0
:SKIP_CFX2COVISESH




IF "%1" NEQ "tecplot2covise" GOTO SKIP_TECPLOT2COVISE
   
   ECHO ...starting tecplot2covise
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%\negotiator\import
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\converters
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%
   SET PATH=%PATH%;%PYTHONPATH%
   
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIBASEDIR%\converters\tecplot2covise.py
   
   SET _DEFAULTCASE=0
:SKIP_TECPLOT2COVISE

IF "%1" NEQ "cfx2covise" GOTO SKIP_CFX2COVISE
   
   ECHO ...starting tecplot2covise
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%\negotiator\import
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\converters
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%
   SET PATH=%PATH%;%PYTHONPATH%
   
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIBASEDIR%\converters\cfx2covise.py
   
   SET _DEFAULTCASE=0
:SKIP_CFX2COVISE

IF "%1" NEQ "ensight2covise" GOTO SKIP_ENSIGHT2COVISESH
   
   ECHO ...starting ensight2covise
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\auxils
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%\negotiator
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%\negotiator\import
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\converters
   SET PYTHONPATH=!PYTHONPATH!;%_UIQT4BASE%
   SET PATH=%PATH%;%PYTHONPATH%
   
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %_UIBASEDIR%\converters\ensight2covise.py
   
   SET _DEFAULTCASE=0
:SKIP_ENSIGHT2COVISESH


IF "%1" NEQ "bifbof2covise.sh" GOTO SKIP_BIFBOF2COVISESH
   
   ECHO ...starting z
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\vr-prepare\converters
   SET PYTHONPATH=!PYTHONPATH!;%_UIBASEDIR%\vr-prepare
   
   %_UNIXSHELL% %_UIBASEDIR%\vr-prepare\converters\bifbof2covise.sh %2 %3 %4 %5 %6 %7 %8 %9
   
   SET _DEFAULTCASE=0
:SKIP_BIFBOF2COVISESH


IF "%1" NEQ "eType" GOTO SKIP_ETYPE
   
   ECHO ...starting alpha
   
   SET PYTHONPATH=%PYTHONPATH%;%_UIBASEDIR%\vr-prepare
   
   %COVISEDIR%\src\application\SCA\GUI\bin\eType %2 %3 %4 %5 %6 %7 %8 %9
   
   SET _DEFAULTCASE=0
:SKIP_ETYPE


IF "%1" NEQ "RealFluid" GOTO SKIP_REALFLUID
   
   ECHO "... starting COVISE for RTT Real Fluid"
   
   SET "PATH=%COVISEDIR%\%ARCHSUFFIX%\lib;%COVISEDIR%\%ARCHSUFFIX%\bin;%EXTERNLIBS%\xerces\lib;%EXTERNLIBS%\QT-4.4.3\lib;%PATH%"
   SET "PYTHONPATH=%COVISEDIR%\Python\bin\RTT\Server\Python;%PYTHONPATH%"
   
   IF NOT [%2]==[] SET "RTT_COVISE_SERVER_PORT=%2"
   
   ECHO RUN
   REM ECHO %PATH%
   CD %COVISEDIR%\Python\bin\RTT\Server
   %COVISEDIR%\%ARCHSUFFIX%\bin\covise --script %COVISEDIR%\Python\bin\RTT\Server\Python\startServer.py
   
   SET _DEFAULTCASE=0
:SKIP_REALFLUID


IF "%_DEFAULTCASE%" NEQ "1" GOTO SKIP_DEFAULTCASE
   
   ECHO ... interpreting argument 1 as binary executable; calling...
   ECHO CALL %COVISEDIR%\%ARCHSUFFIX%\bin\%1 %2 %3 %4 %5 %6 %7 %8 %9
   CALL %COVISEDIR%\%ARCHSUFFIX%\bin\%1 %2 %3 %4 %5 %6 %7 %8 %9
   
   REM for debugging 
   REM todo: find a suitable debugger replacement for ddd
   REM ddd %COVISEDIR%\%ARCHSUFFIX%\bin\%1 %2 %3 %4 %5 %6 %7 %8 %9
      REM valgrind --trace-children=yes --log-file=xxx %COVISEDIR%\%ARCHSUFFIX%\bin\%1
   REM ddd %COVISEDIR%\%ARCHSUFFIX%\bin\IO_Module\ReadEnsightNT %2 %3 %4 %5 %6 %7 %8 %9
   
:SKIP_DEFAULTCASE


ECHO ... done!
GOTO END



:USAGE
ECHO ************************************************************
ECHO *                                                          *
ECHO * This script starts a specified COVISE python script.     *
ECHO *                                                          *
ECHO * Usage:                                                   *
ECHO *    runCOVISEScriptIF                                     *
ECHO *       [name of python script]                            *
ECHO *       [script argument 1]                                *
ECHO *        ...                                               *
ECHO *       [script argument n]                                *
ECHO *                                                          *
ECHO * (C) Copyright 2009 VISENSO GmbH                          *
ECHO *                                                          *
ECHO ************************************************************
ENDLOCAL
EXIT /B 1



:END
ENDLOCAL
EXIT /B 0
