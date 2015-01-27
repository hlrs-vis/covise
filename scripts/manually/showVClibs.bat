@ECHO OFF
ECHO Visual Studio C Runtime Libraries on this system
ECHO are installed in %WINDIR%\WinSxS\
DIR /B %WINDIR%\WinSxS\x86_Microsoft.VC80.CRT_*
PAUSE
