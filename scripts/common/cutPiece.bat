@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author Harry Trautmann
REM *****************************************
REM echos lines of a file
REM startline is determined by startpattern 
REM endline is successing line of line containing endpattern
REM *****************************************

SETLOCAL ENABLEDELAYEDEXPANSION

SET _START=%1
SET _END=%2
SET _FILE=%3
SET _UNIXUTILS=%4

SET _TMPFILE1="%TMP%\~cutPiecetmp1.txt"
SET _TMPFILE2="%TMP%\~cutPiecetmp2.txt"

REM get lineno of startpattern
"%_UNIXUTILS%\grep.exe" -n -e %_START% %_FILE% | "%_UNIXUTILS%\sed.exe" s/:.*://g > %_TMPFILE1%
SET /P _LINESTART=<%_TMPFILE1%
SET /A _LINESTART+=1
"%_UNIXUTILS%\tail.exe" +%_LINESTART%l %_FILE% >%_TMPFILE1%

REM get lineno of endpattern
"%_UNIXUTILS%\grep.exe" -n -e %_END% %_TMPFILE1% | "%_UNIXUTILS%\sed.exe" s/:.*://g > %_TMPFILE2%
SET /P _LINEEND=<%_TMPFILE2%
SET /A _LINEEND-=1
IF "x%_LINEEND%x" EQU "x-1x" (
   REM count total lines of file and subtract lines of excluded head
   REM add 1, since count of lines is 1-based
   "%_UNIXUTILS%\grep.exe" -c -e "" %_FILE% >%_TMPFILE2%
   SET /P _LINEEND=<%_TMPFILE2%
   SET /A _LINEEND-=%_LINESTART%
   SET /A _LINEEND+=1
)

REM ECHO _LINESTART=%_LINESTART%
REM ECHO _LINEEND=%_LINEEND%

"%_UNIXUTILS%\grep.exe" -e %_START% %_FILE% | "%_UNIXUTILS%\sed.exe" s/:.*://g
"%_UNIXUTILS%\head.exe" -%_LINEEND% %_TMPFILE1%
GOTO END

:USAGE
ECHO ...
ECHO echos lines of a file
ECHO startline is determined by startpattern 
ECHO endline is successing line of line containing endpattern
ECHO ...
ECHO Usage: 
ECHO %0 
ECHO    [startpattern -- will be included to result if existing
ECHO    [endpattern] -- excluded from result if existing
ECHO    [file containing patterns]
ECHO    [path to UnixUtils] -- optional, if not given, unix utils 
ECHO       are assumed to be in path
ECHO ...
ECHO called executables:
ECHO     %_UNIXUTILS%\sed.exe
ECHO     %_UNIXUTILS%\grep.exe
ECHO     %_UNIXUTILS%\tail.exe
ECHO     %_UNIXUTILS%\head.exe
ECHO     
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
ENDLOCAL
EXIT /B 1



:END
DEL %_TMPFILE2%
DEL %_TMPFILE1%


ENDLOCAL

EXIT /B 0