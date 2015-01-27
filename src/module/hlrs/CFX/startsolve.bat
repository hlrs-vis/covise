net use j: \\visor\raid\home\visdemo

call parse_ccpnodes %CCP_NODES%
FOR /F "tokens=*" %%i in ('type 00.txt') do SET MY_NODES=%%i
del 00.txt

echo MY_NODES=%MY_NODES%

set CFXROOT=c:\ansys\v110\CFX

set CFX_HOME=%CFXROOT%
rem set CFX_HOME=R:\soft\windows\cfx11sp1\v110\CFX

set CFX5_UNITS_DIR=%CFXROOT%\etc
set PATH=%CFXROOT%\bin\winnt-amd64;%CFXROOT%\bin;%PATH%
set CFX_CCS_SUBMIT_SCRIPT=j:\rechenraum\cfxccs.pl
set _CFX_ENABLE_MSMPI_SM=1

echo cfx5solve -par-dist "%MY_NODES%" %1 %2 %3 %4 %5 %6 %7 %8 %9
echo cfx5solve -par-dist "%MY_NODES%" %1 %2 %3 %4 %5 %6 %7 %8 %9 > startsolve.txt
set CCP >> startsolve.txt

echo where cfx5solve > startsolve.txt
where cfx5solve > startsolve.txt

cfx5solve.exe -v -par-dist "%MY_NODES%" %1 %2 %3 %4 %5 %6 %7 %8 %9