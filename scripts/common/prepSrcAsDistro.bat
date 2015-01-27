@ECHO OFF

REM ***
REM prepares the COVISE sources as distribution,
REM which means that COVISE can be started out of the source directories
REM after preparation
REM ...
REM note that this routine is intended for completely unchanged checked out
REM version of the COVISE sources
REM ***
REM author: Harry Trautmann
REM (C) 2009 Copyright VISENSO GmbH
REM ***

SETLOCAL ENABLEDELAYEDEXPANSION

IF "x%1x" EQU "xx" GOTO USAGE
IF "x%2x" EQU "xx" GOTO USAGE
IF "x%3x" EQU "xx" GOTO USAGE
IF "x%4x" EQU "xx" GOTO USAGE

SET COVISEDIR=%1
SET ARCHSUFFIX=%2
SET EXTERNLIBS=%3
SET UNIXUTILS=%4

SET FILETEMP_LEFT="%TMP%\left.txt"
SET FILETEMP_RIGHT="%TMP%\right.txt"
SET FILETEMP_DIFF="%TMP%\diff.txt"
SET FILETEMP_DIFFCNT="%TMP%\diffcnt.txt"

CALL "%~dp0\..\combinePaths.bat"
CALL common.VISENSO.bat %ARCHSUFFIX% %COVISEDIR%

ECHO Preparing COVISE sources as distribution...
ECHO COVISEDIR=%COVISEDIR%
ECHO ARCHSUFFIX=%ARCHSUFFIX%
ECHO EXTERNLIBS=%EXTERNLIBS%
ECHO UNIXUTILS=%UNIXUTILS%

COPY /V /Y "%~dp0\..\Setup\install\coPyModules.py" %COVISEDIR%\Python\

IF NOT EXIST "%~dp0\..\..\licenses\config-license.xml" GOTO NOLICENSEAVAILABLE
   ECHO ...copying license files...
   COPY /V /Y "%~dp0\..\..\licenses\config-license.xml" %COVISEDIR%\config\
   IF EXIST "%~dp0\..\..\licenses\config.license.xml" COPY /V /Y "%~dp0\..\..\licenses\config.license.xml" %COVISEDIR%\config\
GOTO DONELICENSING
:NOLICENSEAVAILABLE
   ECHO ...did not find license file %~dp0\..\..\licenses\config-license.xml...
:DONELICENSING

IF "x%ARCHSUFFIX:~-3,3%x" EQU "xoptx" GOTO ISOPTVERSION
SET _ARCHSUFFIX_OTHER=%ARCHSUFFIX%opt
SET _FILESUFFIX_MINE=_d
SET _FILESUFFIX_OTHER=
GOTO DONEVERSIONCHECK
:ISOPTVERSION
SET _ARCHSUFFIX_OTHER=%ARCHSUFFIX:~0,-3%
SET _FILESUFFIX_MINE=
SET _FILESUFFIX_OTHER=_d
:DONEVERSIONCHECK

ECHO ...copy python binaries
ECHO    from %COVISEDIR%\%ARCHSUFFIX%\lib
ECHO    to %COVISEDIR%\Python\
ECHO ...
COPY /V /Y %COVISEDIR%\%ARCHSUFFIX%\lib\*%_FILESUFFIX_MINE%.pyd %COVISEDIR%\Python\
COPY /V /Y %COVISEDIR%\%ARCHSUFFIX%\lib\*%_FILESUFFIX_MINE%.py %COVISEDIR%\Python\

ECHO ...copy python binaries
ECHO    from %COVISEDIR%\%_ARCHSUFFIX_OTHER%\lib
ECHO    to %COVISEDIR%\Python\
ECHO ...
COPY /V /Y %COVISEDIR%\%_ARCHSUFFIX_OTHER%\lib\*%_FILESUFFIX_OTHER%.pyd %COVISEDIR%\Python\
COPY /V /Y %COVISEDIR%\%_ARCHSUFFIX_OTHER%\lib\*%_FILESUFFIX_OTHER%.py %COVISEDIR%\Python\

COPY /V /Y %COVISEDIR%\src\sys\ScriptingInterface\covise.py %COVISEDIR%\Python\

ECHO ...generating vr-prepare4 GUI base python classes...
CALL "%~dp0\..\manually\manually_uicVRPrepareGUI.bat" %COVISEDIR%

ECHO ... calling %COVISEDIR%\Python\make_all_for_win32.bat ...
ECHO ... to generate necessary python classes for vr-prepare4 ...
IF NOT EXIST %COVISEDIR%\config\config.license.xml GOTO LICENSECHECKED
   ECHO ... WARNING: %COVISEDIR%\config\config.license.xml not found! 
   ECHO ...    generation of COVISE python classes will likely hang!
:LICENSECHECKED
COPY /Y "%~dp0\..\Autobuild\makeBasiModIgnorelist.txt" %COVISEDIR%\Python\ > NUL
CD /D %COVISEDIR%\Python
CALL %COVISEDIR%\Python\make_all_for_win32.bat

ECHO ... applying temporary vr-prepare4 fixes ...
COPY /V /Y "%~dp0\..\Autobuild\patch_coPyModules.*" %COVISEDIR%\Python\ > NUL
CALL %COVISEDIR%\Python\patch_coPyModules.bat >> %COVISEDIR%\Python\coPyModules.py
DEL /Q %COVISEDIR%\Python\patch_coPyModules.*
IF NOT EXIST "%~dp0\..\Setup\install\coPyModules.py" GOTO DONEINSTALLCOPYMODULES
   REM compare generated classes against a file possibly containing more (i.e. more current?) python classes
   CALL %UNIXUTILS%\grep.exe -e "class " "%~dp0\..\Setup\install\coPyModules.py" > %FILETEMP_LEFT%
   CALL %UNIXUTILS%\grep.exe -e "class " %COVISEDIR%\Python\coPyModules.py > %FILETEMP_RIGHT%
   CALL "%~dp0compareContents.bat" %FILETEMP_LEFT% %FILETEMP_RIGHT% %UNIXUTILS% > %FILETEMP_DIFF%
   CALL %UNIXUTILS%\grep.exe -c -e "" %FILETEMP_DIFF% > %FILETEMP_DIFFCNT%
   SET /P _LINECNT=<%FILETEMP_DIFFCNT%
   IF "x!_LINECNT!x" EQU "x0x" GOTO DONELINEPATCHING
      ECHO ... !_LINECNT! python classes will be patched into generated coPyModules.py:
      TYPE %FILETEMP_DIFF%
      FOR /F "delims=^" %%G IN (%FILETEMP_DIFF%) DO CALL "%~dp0cutPiece.bat" "%%G" "class" "%~dp0\..\Setup\install\coPyModules.py" %UNIXUTILS% >> %COVISEDIR%\Python\coPyModules.py
   :DONELINEPATCHING
   DEL /Q %FILETEMP_DIFF%
   DEL /Q %FILETEMP_LEFT%
   DEL /Q %FILETEMP_RIGHT%
   DEL /Q %FILETEMP_DIFFCNT%
:DONEINSTALLCOPYMODULES


ECHO ...done!

GOTO END

:USAGE
ECHO ...
ECHO prepares the COVISE sources as distribution,
ECHO which means that COVISE can be started out of the source directories
ECHO after preparation
ECHO ...
ECHO note that this routine is intended for completely unchanged checked out
ECHO version of the COVISE sources
ECHO ...
ECHO usage: %0
ECHO    [COVISE sources directory, e. g. D:\COVISE7.0\covise]
ECHO    [ARCHSUFFIX, e. g. vistaopt]
ECHO    [EXTERNLIBS, e. g. c:\vista]
ECHO    [path to UnixUtils]
ECHO ...
ECHO note:
ECHO    Since in this script coPyModules.py is being generated,
ECHO    a valid license is needed. 
ECHO    If the file %~dp0\..\..\licenses\config-license.xml
ECHO    exists then it will be copied to
ECHO    %COVISEDIR%\config\config.license.xml 
ECHO    Otherwise the user should already have copied a valid
ECHO    license there or the generation of coPyModules.py will
ECHO    hang!
ECHO ...
ECHO called batch scripts and executables:
ECHO     ..\combinePaths.bat
ECHO     common.VISENSO.bat
ECHO     %UNIXUTILS%\grep.exe
ECHO     .\manually_uicVRPrepareGUI.bat
ECHO     %COVISEDIR%\Python\make_all_for_win32.bat
ECHO     %COVISEDIR%\Python\patch_coPyModules.bat
ECHO     %~dp0compareContents.bat
ECHO     %~dp0cutPiece.bat
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
ENDLOCAL
EXIT /B 1

:END
ENDLOCAL
EXIT /B 0