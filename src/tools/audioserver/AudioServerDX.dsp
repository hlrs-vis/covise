# Microsoft Developer Studio Project File - Name="AudioServerDX" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Application" 0x0101

CFG=AudioServerDX - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "AudioServerDX.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "AudioServerDX.mak" CFG="AudioServerDX - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "AudioServerDX - Win32 Release" (based on "Win32 (x86) Application")
!MESSAGE "AudioServerDX - Win32 Debug" (based on "Win32 (x86) Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "AudioServerDX - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /MT /W3 /GX /O2 /I "C:\dev\DX81SDK" /I "C:\dev\DX81SDK\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "LOGFILE" /FR /YX /FD /c
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x809 /d "NDEBUG"
# ADD RSC /l 0x809 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 dsound.lib dinput8.lib dxerr8.lib d3dx8dt.lib d3d8.lib d3dxof.lib dxguid.lib winmm.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /machine:I386
# ADD LINK32 dsound.lib dinput8.lib dxerr8.lib d3dx8dt.lib d3d8.lib d3dxof.lib dxguid.lib winmm.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib wsock32.lib comctl32.lib opengl32.lib glu32.lib /nologo /subsystem:windows /machine:I386 /libpath:"c:\dev\DX81SDK\lib"

!ELSEIF  "$(CFG)" == "AudioServerDX - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /I "C:\dev\DX81SDK\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "DEBUG" /D "LOGFILE" /FR /YX /FD /GZ /c
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x809 /d "_DEBUG"
# ADD RSC /l 0x809 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 dsound.lib dinput8.lib dxerr8.lib d3dx8dt.lib d3d8.lib d3dxof.lib dxguid.lib winmm.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /debug /machine:I386 /pdbtype:sept
# ADD LINK32 dsound.lib dinput8.lib dxerr8.lib d3dx8dt.lib d3d8.lib d3dxof.lib dxguid.lib winmm.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib comctl32.lib wsock32.lib opengl32.lib glu32.lib /nologo /subsystem:windows /debug /machine:I386 /pdbtype:sept /libpath:"c:\dev\DX81SDK\lib"

!ENDIF 

# Begin Target

# Name "AudioServerDX - Win32 Release"
# Name "AudioServerDX - Win32 Debug"
# Begin Group "Quellcodedateien"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\as_cache.cpp
# End Source File
# Begin Source File

SOURCE=.\as_clientsocket.cpp
# End Source File
# Begin Source File

SOURCE=.\as_comm.cpp
# End Source File
# Begin Source File

SOURCE=.\as_Control.cpp
# End Source File
# Begin Source File

SOURCE=.\as_opengl.cpp
# End Source File
# Begin Source File

SOURCE=.\as_serversocket.cpp
# End Source File
# Begin Source File

SOURCE=.\as_Sound.cpp
# End Source File
# Begin Source File

SOURCE=.\AudioServerDX.cpp
# End Source File
# Begin Source File

SOURCE=.\common.cpp
# End Source File
# Begin Source File

SOURCE=.\dmutil.cpp
# End Source File
# Begin Source File

SOURCE=.\dsutil.cpp
# End Source File
# Begin Source File

SOURCE=.\dxutil.cpp
# End Source File
# Begin Source File

SOURCE=.\istream.cpp
# End Source File
# Begin Source File

SOURCE=.\loader.cpp
# End Source File
# End Group
# Begin Group "Header-Dateien"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\as_cache.h
# End Source File
# Begin Source File

SOURCE=.\as_client_api.h
# End Source File
# Begin Source File

SOURCE=.\as_clientsocket.h
# End Source File
# Begin Source File

SOURCE=.\as_comm.h
# End Source File
# Begin Source File

SOURCE=.\as_Control.h
# End Source File
# Begin Source File

SOURCE=.\as_opengl.h
# End Source File
# Begin Source File

SOURCE=.\as_serversocket.h
# End Source File
# Begin Source File

SOURCE=.\as_Sound.h
# End Source File
# Begin Source File

SOURCE=.\AudioServerDX.h
# End Source File
# Begin Source File

SOURCE=.\common.h
# End Source File
# Begin Source File

SOURCE=.\dmutil.h
# End Source File
# Begin Source File

SOURCE=.\dsutil.h
# End Source File
# Begin Source File

SOURCE=.\dxutil.h
# End Source File
# Begin Source File

SOURCE=.\loader.h
# End Source File
# Begin Source File

SOURCE=.\precomp.h
# End Source File
# Begin Source File

SOURCE=.\resource.h
# End Source File
# End Group
# Begin Group "Ressourcendateien"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# Begin Source File

SOURCE=.\as16.bmp
# End Source File
# Begin Source File

SOURCE=.\as32.bmp
# End Source File
# Begin Source File

SOURCE=.\as_logo.bmp
# End Source File
# Begin Source File

SOURCE=.\as_logo_mini.bmp
# End Source File
# Begin Source File

SOURCE=.\AudioServer.ico
# End Source File
# Begin Source File

SOURCE=.\AudioServerDX.rc
# End Source File
# Begin Source File

SOURCE=.\bitmap5.bmp
# End Source File
# Begin Source File

SOURCE=.\bounce.wav
# End Source File
# Begin Source File

SOURCE=.\DirectX.ico
# End Source File
# Begin Source File

SOURCE=.\grid.bmp
# End Source File
# Begin Source File

SOURCE=.\init.ico
# End Source File
# Begin Source File

SOURCE=.\notinit.ico
# End Source File
# Begin Source File

SOURCE=.\play.ico
# End Source File
# Begin Source File

SOURCE=.\playloop.ico
# End Source File
# Begin Source File

SOURCE=.\stop.ico
# End Source File
# End Group
# End Target
# End Project
