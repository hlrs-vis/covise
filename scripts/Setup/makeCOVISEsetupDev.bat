@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM ******************************************
REM Sets the environment variables for a 
REM COVISE setup.exe taylored as a VISENSO
REM distribution.
REM ******************************************

SETLOCAL ENABLEDELAYEDEXPANSION

CALL "%~dp0\..\combinePaths.bat"

ECHO Will set environment for VISENSO distribution
ECHO Starting...

IF "x%1x" EQU "xx" GOTO USAGE
IF "x%2x" EQU "xx" GOTO USAGE
IF "x%3x" EQU "xx" GOTO USAGE
IF "x%4x" EQU "xx" GOTO USAGE
IF "x%5x" EQU "xx" GOTO USAGE

SET _COVISESETUPSCRIPT=%1
SET _ARCHSUFFIX=%2
SET _COVSRC=%3
SET _INSTALLTARGET=%4
SET COVISE_DISTRO_TYPE=%5
SET _QUIET=%6

ECHO _INSTALLTARGET=%_INSTALLTARGET%
ECHO _COVSRC=%_COVSRC%
ECHO EXTERNLIBS=%EXTERNLIBS%
ECHO _ARCHSUFFIX=%_ARCHSUFFIX%
ECHO INNOSETUP_HOME=%INNOSETUP_HOME%
ECHO UNIXUTILS_HOME=%UNIXUTILS_HOME%
ECHO COVISE_DISTRO_TYPE=%COVISE_DISTRO_TYPE%
ECHO COVISE_DISTRO_TIMEPREFIX=%COVISE_DISTRO_TIMEPREFIX%
ECHO _QUIET=%_QUIET%

IF NOT EXIST "%~dp0\install\%_INSTALLTARGET%\common.local.bat" GOTO NOCOMMONLOCAL
   ECHO Will include file %~dp0\install\%_INSTALLTARGET%\common.local.bat
   ECHO Please check that its contents are as desired!
   PAUSE
   REM note: copy the file to be installed into a separate folder to avoid
   REM    overwriting the developer´s common.local.bat
   REM    why is a copy needed? because covise.iss does not know about the 
   REM    "%~dp0\install\%_INSTALLTARGET%\" folders
   COPY /Y "%~dp0\install\%_INSTALLTARGET%\common.local.bat" "%_COVSRC%\install"
   GOTO DONECOMMONLOCAL
:NOCOMMONLOCAL
   ECHO Warning: did not find common.local.bat for _INSTALLTARGET=%_INSTALLTARGET%
   ECHO These batch files won´t work without it: 
   ECHO    runOpenCOVER.bat, runCOVISE.bat, runCOVISErdaemon.bat, runVRPrepare4.bat
   PAUSE
:DONECOMMONLOCAL

SET PATH=%PATH%;%INNOSETUP_HOME%

COPY /Y "%~dp0\install\runOpenCOVER.bat" %_COVSRC%\
COPY /Y "%~dp0\install\runCOVISE.bat" %_COVSRC%\
COPY /Y "%~dp0\install\runCOVISErdaemon.bat" %_COVSRC%\
COPY /Y "%~dp0\install\runVRPrepare4.bat" %_COVSRC%\
REM COPY /Y "%~dp0\install\coPyModules.py" %_COVSRC%\Python\
COPY /Y "%~dp0\..\manually\showVClibs.bat" %_COVSRC%\
COPY /Y %_COVSRC%\src\sys\ScriptingInterface\covise.py %_COVSRC%\Python\

ECHO ... setting general COVISE variables ...
CALL common.VISENSO.bat %_ARCHSUFFIX% %_COVSRC%

ECHO ... setting COVISE distribution variables ...
SET COVISE_SIMULATION=NO
SET COVISE_DISTRIBUTION=VISENSO
REM SET FAT_DEVEL=
SET COVISE_GPL_CLEAN=YES
SET COVISE_ARCHSUFFIX=%_ARCHSUFFIX%
SET COVISE_DEVELOPMENT=YES
REM SET COVISE_DIST_DOC= 

IF "x%COVISE_DISTRO_SKIPSOURCEPREP%x" EQU "xx" GOTO SOURCEPREP_DONE
   ECHO ... preparing source as distribution ...
   CALL "%~dp0..\common\prepSrcAsDistro.bat" %_COVSRC% %_ARCHSUFFIX% %EXTERNLIBS% %UNIXUTILS_HOME%
:SOURCEPREP_DONE

ECHO ... executing setup compiler ...
REM change into the COVISE source directory, so that the inno setup compiler 
REM is able to find the included *.iss files
PUSHD .
CD /D %_COVSRC%
REM CALL WHERE iscc.exe > NUL
REM IF ERRORLEVEL 1 GOTO NOSETUPCOMPILER
iscc.exe %_QUIET% %_COVISESETUPSCRIPT%
REM direct call to inno compiler: compil32.exe /cc %_COVISESETUPSCRIPT%
POPD

ECHO ...done!
GOTO END

:NOSETUPCOMPILER
ECHO ...
ECHO ERROR: No setup compiler found!
ECHO    Setup compiler needs to be in the PATH!
ECHO ...
GOTO USAGE

:USAGE
ECHO ...
ECHO usage: %0
ECHO    [absolute path + filename of setup script]
ECHO    [target COVISE archsuffix]
ECHO    [path of COVISE sources]
ECHO    [name of subfolder containing files specific to this distro, e. g. RTT64]
ECHO    [specify distribution type here, currently supported:
ECHO       RTT for COVISE RTT RealFluid distro
ECHO       CC for COVISE CyberClassroom distro
ECHO       anything else will do a plain vanilla ^(i. e. standard^) COVISE distro]
ECHO    [supply /Q for silent compile run]
ECHO ...
ECHO called batch scripts and executables:
ECHO     ..\combinePaths.bat
ECHO     common.VISENSO.bat ^(either in .. or in ..\common^)
ECHO     %INNOSETUP_HOME%\iscc.exe
ECHO     ..\common\prepSrcAsDistro.bat
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
ENDLOCAL
EXIT /B 1

:END
ENDLOCAL
EXIT /B 0