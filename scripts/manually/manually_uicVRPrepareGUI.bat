@ECHO OFF

REM if this script fails, it could be due to hardcoded paths in 
REM PyQt´s pyuic4.bat, which is found in %PYTHONHOME%

SETLOCAL ENABLEDELAYEDEXPANSION

IF "x%1x" EQU "xx" GOTO USAGE

CALL "%~dp0\..\combinePaths.bat"

SET _COVISEDIR=%1

ECHO ... generating Qt ui classes ...
ATTRIB -R %_COVISEDIR%\src\Visenso\branches\pyqt4 /S /D

FOR /F %%G IN ('DIR %_COVISEDIR%\Python\bin\vr-prepare\converters\designerfiles\*.ui /N /B ') DO (
   ECHO now uic'ing: %_COVISEDIR%\Python\bin\vr-prepare\converters\designerfiles\%%G
   SET _TMPPYFILE=%%G
   DEL /Q %_COVISEDIR%\Python\bin\vr-prepare\converters\!_TMPPYFILE:~0,-2!*
   CALL python.exe %_COVISEDIR%\Python\bin\vr-prepare\wqtpyuic4.py -w %_COVISEDIR%\Python\bin\vr-prepare\converters\designerfiles\%%G > %_COVISEDIR%\Python\bin\vr-prepare\converters\%%G
)

FOR /F %%G IN ('DIR %_COVISEDIR%\Python\bin\vr-prepare\designerfiles\*.ui /N /B ') DO (
   ECHO now uic'ing: %_COVISEDIR%\Python\bin\vr-prepare\designerfiles\%%G
   SET _TMPPYFILE=%%G
   DEL /Q %_COVISEDIR%\Python\bin\vr-prepare\!_TMPPYFILE:~0,-2!*
   CALL python.exe %_COVISEDIR%\Python\bin\vr-prepare\wqtpyuic4.py -w %_COVISEDIR%\Python\bin\vr-prepare\designerfiles\%%G > %_COVISEDIR%\Python\bin\vr-prepare\%%G
)

REN %_COVISEDIR%\Python\bin\vr-prepare\*.ui *.py
REN %_COVISEDIR%\Python\bin\vr-prepare\converters\*.ui *.py
IF EXIST %_COVISEDIR%\Python\bin\vr-prepare\*.ui (
   ECHO ... for these Qt ui scripts python scripts already existed:
   DIR %_COVISEDIR%\Python\bin\vr-prepare\*.ui /N /B
   DEL /Q %_COVISEDIR%\Python\bin\vr-prepare\*.ui
)

ECHO ... generating binary resource files
IF EXIST %_COVISEDIR%\Python\bin\vr-prepare\designerfiles\images\StaticImages.qrc rcc.exe -binary %_COVISEDIR%\Python\bin\vr-prepare\designerfiles\images\StaticImages.qrc -o %_COVISEDIR%\Python\bin\vr-prepare\StaticImages.rcc
FOR /F %%G IN ('DIR %_COVISEDIR%\Python\bin\vr-prepare\designerfiles\images\Style_* /N /B ') DO (
   IF EXIST %_COVISEDIR%\Python\bin\vr-prepare\designerfiles\images\%%G\%%G.qrc (
      ECHO now rcc'ing: %_COVISEDIR%\Python\bin\vr-prepare\designerfiles\images\%%G\%%G.qrc
      rcc.exe -binary %_COVISEDIR%\Python\bin\vr-prepare\designerfiles\images\%%G\%%G.qrc -o %_COVISEDIR%\Python\bin\vr-prepare\%%G.rcc
   )
)


ECHO ... compiling Python scripts ...
REM note: only the binary vr-prepare4 is shipped; 
REM    so delete the scripts (including the ui descriptions) after compilation
REM But first fix the unix-hardlinks to avoid confusing errors
DEL /Q %_COVISEDIR%\Python\bin\vr-prepare\VRPUtils.py
COPY /V /Y %_COVISEDIR%\Python\bin\vr-prepare\Utils.py %_COVISEDIR%\Python\bin\vr-prepare\VRPUtils.py
DEL /Q %_COVISEDIR%\Python\bin\vr-prepare\myauxils.py
COPY /V /Y %_COVISEDIR%\Python\bin\vr-prepare\auxils.py %_COVISEDIR%\Python\bin\vr-prepare\myauxils.py
CALL generatePYC.bat %_COVISEDIR%\Python -r -q
CALL generatePYC.bat %_COVISEDIR%\Python\bin\vr-prepare -r -q
CALL generatePYC.bat %_COVISEDIR%\Python\bin\vr-prepare\converters -r -q


ECHO ...done!

GOTO END

:USAGE
ECHO Usage:
ECHO %0
ECHO    [COVISE base directory, e. g. C:\COVISE\covise]
ECHO ...
ECHO generates UI Python classes of COVISE VR-Prepare GUI, compiles them and
ECHO the rest of the VR-Prepare classes
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
GOTO END

:END

ENDLOCAL