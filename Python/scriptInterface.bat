@ECHO OFF
REM #########################################################################
REM #
REM # startup script to overcome restrictions of the interpreter line 
REM # (max length = 30chars)
REM #
REM # UW HLRS, 2008 (C)
REM #
REM #########################################################################

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
        IF /i "%ARCHSUFFIX%" == "tamarauopt" (
          set USE_OPT_LIBS=1
        ) ELSE (
          IF /i "%ARCHSUFFIX%" == "angusopt" (
            set USE_OPT_LIBS=1
          ) ELSE (
	    IF /i "%ARCHSUFFIX%" == "mingw" (
              set USE_OPT_LIBS=1
            ) ELSE (
              set USE_OPT_LIBS=0
	    )
          )
          )
        )
      )
    )
  )
)

IF NOT EXIST "%EXTERNLIBS%"\python\bin\python.exe GOTO nopybin
IF "%USE_OPT_LIBS%" == "1" (
set _PYTHON="%EXTERNLIBS%"\python\bin\python
) ELSE (
set _PYTHON="%EXTERNLIBS%"\python\bin\python_d
)
GOTO doneBin
:nopybin
IF "%USE_OPT_LIBS%" == "1" (
set _PYTHON="%EXTERNLIBS%"\python\python
) ELSE (
set _PYTHON="%EXTERNLIBS%"\python\python_d
)
:doneBin
set _STARTUP="%COVISEDIR%"\Python\scriptInterface.py
set _PYOPT=-i

IF NOT "x%COVISE_LOCAL_PYTHON%x" == "xx" (
   ECHO #    using local python interpreter %COVISE_LOCAL_PYTHON%
   SET _PYTHON=%COVISE_LOCAL_PYTHON%
   SET PYTHONHOME=
)

rem cd %COVISEDIR%\Python

ECHO %_PYTHON% %_PYOPT% %_STARTUP% %1 %2 %3 %4 %5 %6 %7 %8
%_PYTHON% %_PYOPT% %_STARTUP% "%1" "%2" "%3" "%4" "%5" "%6" %7
