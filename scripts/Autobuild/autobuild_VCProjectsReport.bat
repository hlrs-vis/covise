@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM assembles a report containing the errors of 
REM failed builds
REM *******************************************

IF /I "x%1x" EQU "xx" GOTO USAGE
IF /I "x%2x" EQU "xx" GOTO USAGE
IF /I "x%3x" EQU "xx" GOTO USAGE
IF /I "x%4x" NEQ "xx" SET _ERRORATTACHMENT=%4
IF NOT EXIST %2\%1 (
   ECHO Error: No such file found in path!
   ECHO File: %1
   ECHO Path: %2
   GOTO USAGE
)

PUSHD .
CD /D %2
SETLOCAL ENABLEDELAYEDEXPANSION

SET _SED=%3\sed.exe
SET _GREP=%3\grep.exe
SET _RESULTSFILE=%1

SET _TMP0="%TMP%\~tmp0.txt"
SET _TMP1="%TMP%\~tmp1.txt"
SET _TMP2="%TMP%\~tmp2.txt"
SET _TMP3="%TMP%\~tmp3.txt"
SET _TMP4="%TMP%\~tmp4.txt"
SET _TIME1="%TMP%\~time1.txt"
SET _TIME2="%TMP%\~time2.txt"
SET _DATE1="%TMP%\~date1.txt"
SET _DATE2="%TMP%\~date2.txt"

REM construct filename of report
REM strip file suffix and cat current timedate
ECHO %TIME%>%_TIME1%
%_SED% -e "s/,[0-9][0-9]//g" %_TIME1% > %_TIME2%
%_SED% -e "s/://g" %_TIME2% >%_TIME1%
SET /P _TIMEPOSTFIX=<%_TIME1%
DEL %_TIME1%
ECHO %DATE%>%_DATE1%
%_SED% -e "s/\.//g" %_DATE1% >%_DATE2%
SET /P _DATEPOSTFIX=<%_DATE2%
DEL %_DATE1%
DEL %_DATE2%
ECHO %_RESULTSFILE%>%_TMP1%
%_SED% -e "s/\.[a-zA-Z][a-zA-Z][a-zA-Z]//g" %_TMP1%>%_TMP2%
SET /P _REPORTFILE=<%_TMP2%
SET _REPORTFILE=%2\report_%_REPORTFILE%%_DATEPOSTFIX%%_TIMEPOSTFIX%.txt
DEL %_TMP1%
DEL %_TMP2%

IF EXIST %_REPORTFILE% DEL /Q %_REPORTFILE%

REM prepend path to resultsfile
SET _RESULTSFILE=%2\%_RESULTSFILE%

REM gather all summary lines and
REM filter only summary lines containing actual errors
REM then determine, if any processes failed at all
%_GREP% -h -e "error(s)," %_RESULTSFILE% > %_TMP1%
%_GREP% -h -v -e " 0 error(s)," %_TMP1% > %_TMP2%
%_GREP% -h -c -e "" %_TMP2% > %_TMP0%
%_GREP% -h -e " general error " %_RESULTSFILE% >> %_TMP2%
DEL %_TMP1%
SET /P _FAILCOUNT=<%_TMP0%
IF /I "x%_FAILCOUNT%x" EQU "x0x" GOTO NOFAILEDPROJECTS
ECHO Number of failed projects: %_FAILCOUNT%

REM for each failed project print out all errors to the report file
REM first build a file containing the build process number of all failed projects
ECHO Processing build results; please be patient...
COPY %_TMP2% %_TMP3% > NUL
%_SED% -e "s/>.*//g" %_TMP3% > %_TMP4%
SET /P _TIMEWITHOUTMSEC=<%_TIME2%
REM write report header
ECHO -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=- >> %_REPORTFILE%
ECHO -=-=-= BEGIN REPORT OF FAILED BUILD PROCESSES -=-= >> %_REPORTFILE%
ECHO -=-=-=                                      -=-=-= >> %_REPORTFILE%
ECHO -=-=-=                                      -=-=-= >> %_REPORTFILE%
ECHO -=-=-= Report assembled on %_TIMEWITHOUTMSEC% h @ %DATE% >> %_REPORTFILE%
ECHO -=-=-= Resultsfile: %_RESULTSFILE% >> %_REPORTFILE%
ECHO -=-=-= Number of failed projects: %_FAILCOUNT% >> %_REPORTFILE%
ECHO -=-=-=                                      -=-=-= >> %_REPORTFILE%
ECHO -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=- >> %_REPORTFILE%
ECHO -=-=-=                                      -=-=-= >> %_REPORTFILE%
ECHO delete me > %_TMP4%
FOR /F %%G IN ('%_SED% -e "s/>.*//g" %_TMP3%') DO (

   REM get all messages of current build process number
   %_GREP% -h -e "^%%G>" %_RESULTSFILE% > %_TMP4%

   REM find out if build process failed
   %_GREP% -h -c -e " 0 error(s)" %_TMP4% > %_TMP0%
   SET /P _BUILDFAIL=<%_TMP0%
   DEL %_TMP0%
   IF "x!_BUILDFAIL!x" EQU "x0x" (

      REM print project name of current build process number
      %_GREP% -h -e "^%%G>------" %_RESULTSFILE%>>%_REPORTFILE%
   
      REM print all errors of current build process number
      %_GREP% -h -e "error C" %_TMP4%>>%_REPORTFILE%
      %_GREP% -h -e "error LNK" %_TMP4%>>%_REPORTFILE%

      REM print error summary of current build process number
      %_GREP% -h -e "^%%G>" %_TMP2%>>%_REPORTFILE%

      ECHO -=-=-=                                      -=-=-=>>%_REPORTFILE%
      ECHO -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=->>%_REPORTFILE%
      ECHO -=-=-=                                      -=-=-=>>%_REPORTFILE%
   )
)
DEL %_TMP4%
ECHO -=-=-=                                      -=-=-=>>%_REPORTFILE%
ECHO -=-=-= END REPORT OF FAILED BUILD PROCESSES -=-=-=>>%_REPORTFILE%
ECHO -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=->>%_REPORTFILE%

REM copy output file to the file named in environment variable %ERRORATTACHMENT%
REM which is to be created by the autobuild process
IF DEFINED _ERRORATTACHMENT (
   REM IF EXIST %ERRORATTACHMENT% DEL /Q %ERRORATTACHMENT%
   REM COPY /V /Y %_REPORTFILE% %ERRORATTACHMENT% > NUL
   REM instead of copying the report file´s content, append it
   TYPE %_REPORTFILE%>>%_ERRORATTACHMENT%
)

DEL %_TMP2%
DEL %_TMP3%
ECHO ...done!
GOTO END

:USAGE
ECHO ...
ECHO Usage: %0 
ECHO    [name of file with build process results]
ECHO    [path to results file]
ECHO    [path to unix tools sed and grep]
ECHO    [path + filename to have the report file´s content appended]
ECHO ...
ECHO called executables:
ECHO     sed.exe
ECHO     grep.exe
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
ENDLOCAL
POPD
EXIT /B 1

:NOFAILEDPROJECTS
ECHO No build processes failed
ECHO ...done!
GOTO END

:END
IF EXIST %_TMP0% DEL /Q %_TMP0%
IF EXIST %_TMP1% DEL /Q %_TMP1%
IF EXIST %_TMP2% DEL /Q %_TMP2%
IF EXIST %_TMP3% DEL /Q %_TMP3%
IF EXIST %_TMP4% DEL /Q %_TMP4%
IF EXIST %_TIME1% DEL /Q %_TIME1%
IF EXIST %_TIME2% DEL /Q %_TIME2%
IF EXIST %_DATE1% DEL /Q %_DATE1%
IF EXIST %_DATE2% DEL /Q %_DATE2%
ENDLOCAL
POPD
EXIT /B 0