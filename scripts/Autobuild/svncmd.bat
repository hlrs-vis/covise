@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM ******************************************
REM This script shells the svn calls in a way that an 
REM individual SVN_SSH environment has to be supplied.
REM This script is intended to be used inside a build
REM environment like ant.
REM ******************************************

IF [%1] EQU [] GOTO USAGE
IF [%2] EQU [] GOTO USAGE
IF [%3] EQU [] GOTO USAGE

ECHO svn shelled command ...
SET SVN_SSH=%1 -l %2
ECHO ... SVN_SSH=%SVN_SSH%
ECHO ... calling svn %3 %4 %5 %6 %7 %8 %9
svn %3 %4 %5 %6 %7 %8 %9
ECHO ...done!

GOTO END

:USAGE
ECHO ...
ECHO %0 
ECHO    [path of plink tool, e. g. C:\\Programme\\TortoiseSVN\\bin\\TortoisePlink.exe]
ECHO    [svn user]
ECHO    [svn command]
ECHO    [...further svn parameters here...]
ECHO ...
ECHO note: environment variable SVN_SSH is set to 
ECHO     firstParameter -l secondParameter
ECHO     Also note that no password is supplied with the svn command. 
ECHO     This is due to a setup authentication agent like PuTTY´s pageant. 
ECHO     You have to do that beforehand also.
ECHO ...
ECHO (C) Copyright 2009 VISENSO GmbH
ECHO ...
EXIT /B 1

:END
EXIT /B 0