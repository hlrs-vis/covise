@ECHO OFF
SETLOCAL

SET _UNIXUTILS=c:\Vista\UnixUtils
SET _VCRESULTFILE=VCbuild_temp.txt

CALL "%~dp0\..\combinePaths.bat"
CALL autobuild_VCProjectsReport.bat %_VCRESULTFILE% %~dp0 %_UNIXUTILS% "%TMP%\deleteme.txt"
DEL /Q "%TMP%\deleteme.txt"

ENDLOCAL