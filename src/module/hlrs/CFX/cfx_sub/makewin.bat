C:\Intel\Compiler\Fortran\9.1\EM64T\Bin\IFortVars.bat
%VSINSTALLDIR%\VC\vcvarsall.bat amd64
set CFXDIR=R:\soft\windows\cfx11sp1\v110\CFX

ifort -Zi -iface:cvf -MD -fpp -include:%CFXDIR%\include -object:winnt-amd64\user_covise.o -c user_covise.F
ifort -Zi -iface:cvf -MD -fpp -include:%CFXDIR%\include -object:winnt-amd64\user_export.o -c user_export.F
%VSINSTALLDIR%\VC\bin\amd64\cl.exe -Zi -MD -DWIN32 -DMIXED_STR_LEN -I%CFXDIR%\include -Fowinnt-amd64\coSimClient.o -c coSimClient.c
%VSINSTALLDIR%\VC\bin\amd64\link.exe /debug /DLL /out:winnt-amd64\user_export_mpich2.dll winnt-amd64\user_covise.o winnt-amd64\user_export.o winnt-amd64\coSimClient.o %CFXDIR%\lib\winnt-amd64\solver-mpich2.lib %VSINSTALLDIR%\VC\PlatformSDK\Lib\AMD64\WS2_32.Lib kernel32.lib
%VSINSTALLDIR%\VC\bin\amd64\link.exe /debug /DLL /out:winnt-amd64\user_export_pvm.dll winnt-amd64\user_covise.o winnt-amd64\user_export.o winnt-amd64\coSimClient.o %CFXDIR%\lib\winnt-amd64\solver-pvm.lib %VSINSTALLDIR%\VC\PlatformSDK\Lib\AMD64\WS2_32.Lib kernel32.lib
mt.exe /manifest winnt-amd64\user_export_mpich2.dll.manifest /outputresource:"winnt-amd64\user_export_mpich2.dll;2"
mt.exe /manifest winnt-amd64\user_export_pvm.dll.manifest /outputresource:"winnt-amd64\user_export_pvm.dll;2"

%VSINSTALLDIR%\VC\bin\amd64\cl.exe -Zi -MD -DWIN32 -I%CFXDIR%\include -Fowinnt-amd64\user_import.o user_import.c /link  /debug /NODEFAULTLIB:libc.lib /nodefaultlib:libcd.lib %CFXDIR%\lib\winnt-amd64\libmeshimport.lib %CFXDIR%\lib\winnt-amd64\libio.lib .\winnt-amd64\coSimClient.o C:\MSDEV\VC\PlatformSDK\Lib\AMD64\WS2_32.Lib %CFXDIR%\lib\winnt-amd64\libratlas_api.lib %CFXDIR%\lib\winnt-amd64\libratlas.lib %CFXDIR%\lib\winnt-amd64\libcclapilt.lib /out:winnt-amd64\user_import.exe
mt.exe /manifest winnt-amd64\user_import.exe.manifest /outputresource:"winnt-amd64\user_import.exe;1"

del user_import.exe.manifest
