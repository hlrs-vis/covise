@echo off
set COCONFIG=%2
shift
shift
set "args="
:parse
if "%~1" neq "" (
  set args=%args% %1
  shift
  goto :parse
)
if defined args set args=%args:~1%
START /B C:\Progra~1\COVISE\zebuopt\bin\Renderer\OpenCOVER %args%