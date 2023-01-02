@echo off
for /f "tokens=*" %%a in (%0\..\covise.env) do (
  set %%a
)
echo %*
@echo off
setlocal ENABLEDELAYEDEXPANSION
@REM set last=0
@REM for %%x in (%*) do (
@REM    if "%%x" EQU "-DCMAKE_BUILD_TYPE:STRING" (
@REM     set last=1
@REM    )
@REM    IF !last! == 1 (
@REM       if "%%x" EQU "Debug" (
@REM         set ARCHSUFFIX=zebu
@REM       ) else if "%%x" EQU "Release" (
@REM         set ARCHSUFFIX=zebuopt
@REM       ) else if "%%x" EQU "RelWithDebInfo" (
@REM         set ARCHSUFFIX=zebuopt
@REM       ) else if "%%x" EQU "MinSizeRel" (
@REM         set ARCHSUFFIX=zebuopt
@REM       )
@REM    )
@REM )

set BUILD=0
for %%x in (%*) do (
   if "%%x" EQU "--build" set BUILD=1
)

if !BUILD!==1 (
  cmake %* 

) else cmake %* %COVISE_CMAKE_OPTIONS%




 