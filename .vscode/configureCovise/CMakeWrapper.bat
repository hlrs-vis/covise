@echo off
for /f "tokens=*" %%a in (%0\..\covise.env) do (
  set %%a
)
echo %*
@echo off
set last=0
setlocal ENABLEDELAYEDEXPANSION
for %%x in (%*) do (
   if "%%x" EQU "-DCMAKE_BUILD_TYPE:STRING" (
    set last=1
   )
   IF !last! == 1 (
      if "%%x" EQU "Debug" (
        set ARCHSUFFIX=zebu
      ) else if "%%x" EQU "Release" (
        set ARCHSUFFIX=zebuopt
      ) else if "%%x" EQU "RelWithDebInfo" (
        set ARCHSUFFIX=zebuopt
      ) else if "%%x" EQU "MinSizeRel" (
        set ARCHSUFFIX=zebuopt
      )
   )
)

cmake %*


 