REM this script puts all scripts into the path variable
REM assure, that its call happens inside of a SETLOCAL-ENDLOCAL block

REM get the 8.3 filename of current installation
SET TMPFILE="%TMP%\~covtmpfile.txt"
REM note: the following line could produce some cross-grained errors when
REM    %~dp0 contains round parentheses because IF interprets them wrong.
REM    If you have serious problems with this script, a solution could be to
REM    store the current path by PUSHD ., CD /D "%~dp0..", there looking, if 
REM    get8dot3path.vbs is there and setting the VBS8DOT3PATH variable 
REM    accordingly and then POPD to get back.
SET VBS8DOT3PATH="%~dp0\..\get8dot3path.vbs"
IF NOT EXIST %VBS8DOT3PATH% SET VBS8DOT3PATH=%~dp0\..\common\get8dot3path.vbs
CALL "%WINDIR%\system32\cscript.exe" /nologo %VBS8DOT3PATH% "%~dp0">%TMPFILE%
SET /P INSTDIR=<%TMPFILE%
DEL /Q %TMPFILE%

SET PATH=%INSTDIR%\..;%INSTDIR%\Autobuild;%INSTDIR%\common;%INSTDIR%\Setup;%PATH%