; files related to RTT distro

Source: {#COVISEDIR}\RealFluid.bat; DestDir: {app}\covise; Components: deltagen
Source: {#COVISEDIR}\get8dot3path.vbs; DestDir: {app}\covise; Components: deltagen

Source: {#COVISEDIR}\Python\bin\RTT\Server\*.sh; DestDir: {app}\covise\Python\bin\RTT\Server; Components: deltagen
Source: {#COVISEDIR}\Python\bin\RTT\Server\Python\*.pyc; DestDir: {app}\covise\Python\bin\RTT\Server\Python; Components: deltagen
Source: {#COVISEDIR}\Python\bin\RTT\Server\Python\startServer.py; DestDir: {app}\covise\Python\bin\RTT\Server\Python; Components: deltagen
Source: {#COVISEDIR}\Python\bin\RTT\Server\Models\msport\*; DestDir: {app}\covise\Python\bin\RTT\Server\Models\msport; Components: deltagen

Source: {#BIN}\Unsupported\RWCoviseBlock.exe; DestDir: {#DBIN}\Unsupported; Components: deltagen
Source: {#BIN}\IO\ReadEnsight.exe; DestDir: {#DBIN}\IO; Components: deltagen


; *********************************
; configs
Source: {#COVISEDIR}\config\config*.xml; DestDir: {app}\covise\config; Excludes: config.license.xml,config-license.xml; Components: deltagen; Permissions: everyone-full users-full
Source: {#COVISEDIR}\config\config-license.xml; DestDir: {app}\covise\config; DestName: config.license.xml; Components: deltagen; Permissions: everyone-full users-full; Flags: onlyifdoesntexist
