
@ echo off

SETLOCAL ENABLEDELAYEDEXPANSION

REM #########################################################################
REM # this file substitute the Makefile for windows
REM #########################################################################


ECHO ...copy python binaries
ECHO    from %COVISEDIR%\%ARCHSUFFIX%\lib
ECHO    to %COVISEDIR%\Python\
ECHO ...
COPY /V /Y %COVISEDIR%\%ARCHSUFFIX%\lib\*%_FILESUFFIX_MINE%.pyd %COVISEDIR%\Python\
COPY /V /Y %COVISEDIR%\%ARCHSUFFIX%\lib\*%_FILESUFFIX_MINE%.py %COVISEDIR%\Python\

COPY /V /Y %COVISEDIR%\src\sys\ScriptingInterface\covise.py %COVISEDIR%\Python\
COPY /V /Y %COVISEDIR%\build.tamarau\src\sys\ScriptingInterface\covise.py %COVISEDIR%\Python\

COPY /V /Y %COVISEDIR%\src\sys\GuiRenderMessage\coGrmsg.py %COVISEDIR%\Python\
COPY /V /Y %COVISEDIR%\build.tamarau\src\sys\GuiRenderMessage\coGrmsg.py %COVISEDIR%\Python\

rem copy python startup script to the bin directory
rem this script is started by the crb

copy scriptInterface.bat  %COVISEDIR%\%ARCHSUFFIX%\bin\scriptInterface.bat

echo scriptInterface.bat copied to %COVISEDIR%\%ARCHSUFFIX%\bin


rem generate the static python representation of all covise module
rem find errors in the stub file
rem some modules can't be converted under windows 

echo will create stubs for all modules
echo not all will have success, be patient ....
IF /i "%ARCHSUFFIX%" == "win32opt" (
  set USE_OPT_LIBS=1
) ELSE (
  IF /i "%ARCHSUFFIX%" == "vistaopt" (
    set USE_OPT_LIBS=1
  ) ELSE (
    IF /i "%ARCHSUFFIX%" == "amdwin64opt" (
      set USE_OPT_LIBS=1
    ) ELSE (
      IF /i "%ARCHSUFFIX%" == "zackelopt" (
        set USE_OPT_LIBS=1
      ) ELSE (
        IF /i "%ARCHSUFFIX%" == "berrendaopt" (
          set USE_OPT_LIBS=1
        ) ELSE (
          IF /i "%ARCHSUFFIX%" == "angusopt" (
            set USE_OPT_LIBS=1
          ) ELSE (
            set USE_OPT_LIBS=0
          )
        )
      )
    )
  )
)

IF NOT EXIST %EXTERNLIBS%\python\bin\python.exe GOTO nopybin
IF "%USE_OPT_LIBS%" == "1" (
set _PYTHON=%EXTERNLIBS%\python\bin\python 
) ELSE (
set _PYTHON=%EXTERNLIBS%\python\bin\python_d
)
GOTO doneBin
:nopybin
IF "%USE_OPT_LIBS%" == "1" (
set _PYTHON=%EXTERNLIBS%\python\python 
) ELSE (
set _PYTHON=%EXTERNLIBS%\python\python_d
)
:doneBin
IF DEFINED COVISE_LOCAL_PYTHON (
   ECHO using local python interpreter %COVISE_LOCAL_PYTHON%
   SET _PYTHON=%COVISE_LOCAL_PYTHON%
)

SET _MAKEBASIIGNORELIST=
IF EXIST %COVISEDIR%\Python\makeBasiModIgnorelist.txt (
   FOR /F %%G IN (%COVISEDIR%\Python\makeBasiModIgnorelist.txt) DO (
      SET _MAKEBASIIGNORELIST=!_MAKEBASIIGNORELIST! -i%%G
   )
)
ECHO %_PYTHON% %COVISEDIR%\Python\makeBasiModCode.py %_MAKEBASIIGNORELIST%
%_PYTHON% %COVISEDIR%\Python\makeBasiModCode.py %_MAKEBASIIGNORELIST% > coPyModules.py
echo finished !!

ENDLOCAL