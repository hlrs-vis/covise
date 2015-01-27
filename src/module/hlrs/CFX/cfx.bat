@echo off

set CFX_HOME=c:\Ansys\v110\CFX
rem set CFX_HOME=C:\"Program Files\ANSYS Inc"\v110\CFX
set CFX5_UNITS_DIR=
set CFX5_UNITS_DIR=%CFX_HOME%\etc
set PATH=%CFX_HOME%\bin;%PATH%
set PATH=r:\soft\UnixUtils;%PATH%

set LM_LICENSE_FILE=50030@bond.hlrs.de

rem example command line arguments
rem numProc 1 Hostlist viscluster09*1 MachineType radial_machine startup \"serial\" revolutions 1500 deffile 0 maxIterations 100 locationString \\\"Primitive\\ 3D\\ A,Primitive\\ 3D\\ B,Primitive\\ 3D\\ C,Primitive\\ 3D\\ D,Primitive\\ 3D\\ E\\\" CO_SIMLIB_CONN C:141.58.8.22/31500_60.000000_0

set numargs=0

:countem
set arg=%1
if defined arg (
   set /a numargs+=2

   if "%1" == "numProc" (
     set numProc=%2
   )

   if "%1" == "Hostlist" (
     set hostlist=%2
   )

   if "%1" == "CO_SIMLIB_CONN" (
     set CO_SIMLIB_CONN=%2
   )

   if "%1" == "MachineType" (
     set machineType=%2
   )

   if "%1" == "startup" (
     set startup=%2
   )

   if "%1" == "revolutions" (
     set revolutions=%2
   )

   if "%1" == "deffile" (
     set deffile=%2
   )

   if "%1" == "maxIterations" (
     set maxIterations=%2
   )

   if "%1" == "locationString" (
     set locationString=%2
   )


   shift
   shift

   goto countem
)

echo %numargs% arguments


set path=r:\soft\UnixUtils;%PATH%

rem it is ridiculous, but this is equivalent to the bash command locationString=`echo ${locationString} | sed -e "s/\"//g"` 
for /f "usebackq tokens=1* delims=" %%b in (`set locationString ^| sed -e "s/\\//g"`) do set %%b

echo numProc=%numProc%
echo Hostlist=%hostlist%
echo MachineType=%machineType%
echo startup=%startup%
echo revolutions=%revolutions%
echo deffile=%deffile%
echo maxIterations=%maxIterations%
echo locationString=%locationString%
echo CO_SIMLIB_CONN: %CO_SIMLIB_CONN%

rem taskkill /F /IM cfx5solve.exe

rem delete unneeded RadialRunner files
del /Q radialrunner_* 2> NUL
del /Q rr_meridian_* 2> NUL
del /Q rr_gridparams* 2> NUL
del /Q RadialRunner.deb 2> NUL
del /Q MERIDIANelems* 2> NUL
del /Q MERIDIANnodes_* 2> NUL
del /Q rr_blnodes_* 2> NUL

rem delete unneeded CFX files
del /Q radialrunner.def 2> NUL
del /Q radialrunner.def.lck 2> NUL
del /Q radialrunner.def.gtm 2> NUL

set ending1=session_template.pre
set ending2=session.pre
set concat1=%machineType%%ending1%
set concat2=%machineType%%ending2%
echo concat1=%concat1%
echo concat2=%concat2%

for /f "usebackq tokens=1* delims=" %%b in (`set locationString ^| sed -e "s/\"/\\\\\"/g"`) do set %%b

rem set CFXDIR=c:\tmp\rechenraum
net use j: \\visor\raid\home\visdemo
set CFXDIR=j:\rechenraum
set BACKUPCFXDIR=%CFXDIR%

rem bring CFXDIR it in a form for sed ...
rem for /f "usebackq tokens=1* delims=" %%b in (`set CFXDIR ^| sed -e "s/\\/\\\//g"`) do set %%b
for /f "usebackq tokens=1* delims=" %%b in (`set CFXDIR ^| sed -e "s/\\/\\\\/g"`) do set %%b
set SEDCFXDIR=%CFXDIR%
set CFXDIR=%BACKUPCFXDIR%
set CFXDIR_BAK=

echo SEDCFXDIR=%SEDCFXDIR%
echo CFXDIR=%CFXDIR%

cd /D %CFXDIR%

if "%machineType%" == "radial" (
   goto comm1
)
if "%machineType%" == "axial" (
   goto comm1
)
if "%machineType%" == "rechenraum" (
   goto comm1
)
if "%machineType%" == "surfacedemo" (
   goto comm1
)
if "%machineType%" == "radial_machine" (
   goto comm2
)
if "%machineType%" == "axial_machine" (
   goto comm2
)

echo machineType: %machineType%

goto jumpover

:comm1
   echo cat %concat1% | sed -e "s/\$(CFXDIR)/%SEDCFXDIR%/g" -e "s/REVOLUTIONS/%revolutions%/g" -e "s/MAXITERATIONS/%maxIterations%/g" > %concat2%
   cat %concat1% | sed -e "s/\$(CFXDIR)/%SEDCFXDIR%/g" -e "s/REVOLUTIONS/%revolutions%/g" -e "s/MAXITERATIONS/%maxIterations%/g" > %concat2%
   goto jumpover

:comm2
   cat %concat1% | sed -e "s/$(CFXDIR)/%SEDCFXDIR%/g" -e "s/REVOLUTIONS/%revolutions%/g" -e "s/MAXITERATIONS/%maxIterations%/g" -e "s/LOCATION/%locationString%/" > %concat2%

:jumpover

cd /p %CFXDIR%\cfx_sub
touch ghostmesh.gtm
cd %CFXDIR%

job new /numprocessors:%numProc% > 00.txt
set JOBID=cat 00.txt | cut -f4 -d " "
for /f "usebackq tokens=1* delims=" %%b in (`cat 00.txt ^| cut -f4 -d " "`) do set JOBID=%%b
del 00.txt

echo JOBID=%JOBID%

if "%deffile%" == "0" (
   echo executing CFX Pre, receiving mesh and bcs from Covise, writing def file 

   rem echo cfx5pre -batch %CFXDIR%\%concat2%
   rem cfx5pre -batch %CFXDIR%\%concat2%

   rem echo job submit /user:vis08\visdemo /scheduler:141.58.8.215 \\visor\raid\soft\windows\cfx11sp1\v110\CFX\bin\cfx5pre -batch \\vis08-28\c$\tmp\rechenraum\%concat2%
   rem job submit /user:vis08\visdemo /scheduler:141.58.8.215 \\visor\raid\soft\windows\cfx11sp1\v110\CFX\bin\cfx5pre -batch \\vis08-28\c$\tmp\rechenraum\%concat2%

   echo job add %JOBID% /scheduler:141.58.8.215 /workdir:\\visor\raid\home\visdemo\rechenraum /exclusive:true /numcores:1 /env:CO_SIMLIB_CONN=%CO_SIMLIB_CONN% /name:cfxpre \\visor\raid\home\visdemo\rechenraum\startpre.bat -batch %concat2%
   job add %JOBID% /scheduler:141.58.8.215 /workdir:\\visor\raid\home\visdemo\rechenraum /exclusive:true /numcores:1 /env:CO_SIMLIB_CONN=%CO_SIMLIB_CONN%;LM_LICENSE_FILE=50030@bond.hlrs.de /name:cfxpre \\visor\raid\home\visdemo\rechenraum\startpre.bat -batch %concat2%
)

echo starting up solver

cd %CFXDIR%/cfx_sub
rem rm ghostmesh.gtm
cd ..

rem remove "\" character from startup
for /f "usebackq tokens=1* delims=" %%b in (`set startup ^| sed -e "s/\\//g"`) do set %%b
echo startup=%startup%

if %startup% == "MPICH Local Parallel for Windows" (
    echo taking hostlist from scheduler
    set cfxarg=-part %numProc% -start-method
    set cfxstartup="MPICH Local Parallel for Windows"
)
 
if %startup% == "MPICH Distributed Parallel for Windows" (
    echo taking hostlist from scheduler
    set cfxarg=-part %numProc% -start-method
    set cfxstartup="MPICH Distributed Parallel for Windows"
    rem set cfxstartup="PVM Distributed Parallel"
)

if %startup% == "MPICH2 Local Parallel for Windows" (
    echo hostlist is needed here - to be reimplemented
    set cfxarg=-part %numProc% -start-method
    set cfxstartup="MPICH2 Local Parallel for Windows"
)
 
if %startup% == "MPICH2 Distributed Parallel for Windows" (
    echo taking hostlist from scheduler
    set cfxarg=-part %numProc% -start-method
    set cfxstartup="MPICH2 Distributed Parallel for Windows"
)

if %startup% == "MSMPI" (
    set cfxarg=-part %numProc% -start-method
    set cfxstartup=MSMPI
)

if %startup% == "Submit to Windows CCS Queue" (
    set cfxarg=-part %numProc% -start-method
    set cfxstartup="Submit to Windows CCS Queue"
)


if %startup% == "serial" (
    set cfxarg=""
    set cfxstartup=-serial
)

echo deffile="%deffile%"
echo cfxarg=%cfxarg%
echo cfxstartup=%cfxstartup%

if "%deffile%" == "0" (
  echo cfx5solve -def %CFXDIR%/%machineType%.def %cfxarg% %cfxstartup%
  if %startup%=="serial" (
     rem echo cfx5solve -def %CFXDIR%/%machineType%.def %cfxstartup%
     echo job add %JOBID% /scheduler:141.58.8.215 /workdir:\\visor\raid\home\visdemo\rechenraum /exclusive:true /numcores:%numProc% /env:CO_SIMLIB_CONN=%CO_SIMLIB_CONN%;LM_LICENSE_FILE=50030@bond.hlrs.de /name:cfxsolve \\visor\raid\home\visdemo\rechenraum\startsolve.bat -def %machineType%.def %cfxstartup%
     job add %JOBID% /scheduler:141.58.8.215 /workdir:\\visor\raid\home\visdemo\rechenraum /exclusive:true /numcores:%numProc% /env:CO_SIMLIB_CONN=%CO_SIMLIB_CONN%;LM_LICENSE_FILE=50030@bond.hlrs.de /name:cfxsolve \\visor\raid\home\visdemo\rechenraum\startsolve.bat -def %machineType%.def %cfxstartup%
  ) else (
     rem echo cfx5solve -def %CFXDIR%/%machineType%.def %cfxarg% %cfxstartup%
     echo job add %JOBID% /scheduler:141.58.8.215 /workdir:\\visor\raid\home\visdemo\rechenraum /exclusive:true /numcores:%numProc% /env:CO_SIMLIB_CONN=%CO_SIMLIB_CONN%;LM_LICENSE_FILE=50030@bond.hlrs.de /name:cfxsolve \\visor\raid\home\visdemo\rechenraum\startsolve.bat -def %machineType%.def %cfxarg% %cfxstartup%
     job add %JOBID% /scheduler:141.58.8.215 /workdir:\\visor\raid\home\visdemo\rechenraum /exclusive:true /numcores:%numProc% /env:CO_SIMLIB_CONN=%CO_SIMLIB_CONN%;LM_LICENSE_FILE=50030@bond.hlrs.de /name:cfxsolve \\visor\raid\home\visdemo\rechenraum\startsolve.bat -def \\visor\raid\home\visdemo\rechenraum\%machineType%.def %cfxarg% %cfxstartup%
  )

) else (
  rem echo cfx5solve -def %deffile% %cfxarg% %startup%
  echo job add %JOBID% /scheduler:141.58.8.215 /workdir:\\visor\raid\home\visdemo\rechenraum /exclusive:true /numcores:%numProc% /env:CO_SIMLIB_CONN=%CO_SIMLIB_CONN%;LM_LICENSE_FILE=50030@bond.hlrs.de /name:cfxsolve \\visor\raid\home\visdemo\rechenraum\startsolve.bat -def %deffile% %cfxarg% %startup%
  job add %JOBID% /scheduler:141.58.8.215 /workdir:\\visor\raid\home\visdemo\rechenraum /exclusive:true /numcores:%numProc% /env:CO_SIMLIB_CONN=%CO_SIMLIB_CONN%;LM_LICENSE_FILE=50030@bond.hlrs.de /name:cfxsolve \\visor\raid\home\visdemo\rechenraum\startsolve.bat -def %deffile% %cfxarg% %startup%
)

echo job submit /ID:%JOBID%
job submit /ID:%JOBID%

rem net use j: /delete /YES

echo END of cfx.bat - job %JOBID% submitted!