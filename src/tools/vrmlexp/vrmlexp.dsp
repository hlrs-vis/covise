# Microsoft Developer Studio Project File - Name="vrmlexp" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

CFG=vrmlexp - Win32 Release
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "vrmlexp.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "vrmlexp.mak" CFG="vrmlexp - Win32 Release"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "vrmlexp - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "vrmlexp - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "vrmlexp - Win32 Hybrid" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "vrmlexp - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir ".\Release"
# PROP BASE Intermediate_Dir ".\Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir ".\Release"
# PROP Intermediate_Dir ".\Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /YX /c
# ADD CPP /nologo /G6 /MD /W3 /GX /O2 /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "NDEBUG" /D "_LEC_" /D "WIN32" /D "_WINDOWS" /Yu"vrml.h" /Fd"vrmlexp.pdb" /FD /c
# SUBTRACT CPP /Z<none> /Fr
# ADD BASE MTL /nologo /D "NDEBUG" /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /i "..\..\..\include" /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:windows /dll /machine:I386
# ADD LINK32 comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib version.lib ..\..\..\lib\maxscrpt.lib /nologo /base:"0x641d0000" /subsystem:windows /dll /machine:I386 /out:"..\..\..\..\maxsdk\plugin\vrmlexp.dle" /release
# SUBTRACT LINK32 /pdb:none /debug

!ELSEIF  "$(CFG)" == "vrmlexp - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir ".\vrmlexp__"
# PROP BASE Intermediate_Dir ".\vrmlexp__"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir ".\Debug"
# PROP Intermediate_Dir ".\Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /Zi /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /YX /Fd"vrmlexp.pdb" /c
# ADD CPP /nologo /G6 /MDd /W3 /Gm /GX /ZI /Od /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_LEC_" /Yu"vrml.h" /Fd"vrmlexp.pdb" /FD /c
# SUBTRACT CPP /Fr
# ADD BASE MTL /nologo /D "_DEBUG" /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /i "..\..\..\include" /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:windows /dll /debug /machine:I386
# ADD LINK32 comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib version.lib ..\..\..\lib\maxscrpt.lib /nologo /base:"0x641d0000" /subsystem:windows /dll /debug /machine:I386 /out:"..\..\..\..\maxsdk\plugin\vrmlexp.dle"
# SUBTRACT LINK32 /pdb:none /incremental:no

!ELSEIF  "$(CFG)" == "vrmlexp - Win32 Hybrid"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir ".\vrmlexp_0"
# PROP BASE Intermediate_Dir ".\vrmlexp_0"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir ".\Hybrid"
# PROP Intermediate_Dir ".\Hybrid"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /G5 /MD /W3 /Gm /GX /Zi /Od /I "..\..\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /Fd"vrmlexp.pdb" /c
# SUBTRACT BASE CPP /YX
# ADD CPP /nologo /G6 /MD /W3 /Gm /GX /ZI /Od /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_LEC_" /Yu"vrml.h" /Fd"vrmlexp.pdb" /FD /c
# SUBTRACT CPP /Fr
# ADD BASE MTL /nologo /D "_DEBUG" /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /i "..\..\..\include" /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:windows /dll /debug /machine:I386 /out:"..\..\..\maxsdk\plugin\vrmlexp.dle"
# ADD LINK32 comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib version.lib ..\..\..\lib\maxscrpt.lib /nologo /base:"0x641d0000" /subsystem:windows /dll /debug /machine:I386 /out:"..\..\..\..\maxsdk\plugin\vrmlexp.dle"

!ENDIF 

# Begin Target

# Name "vrmlexp - Win32 Release"
# Name "vrmlexp - Win32 Debug"
# Name "vrmlexp - Win32 Hybrid"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;hpj;bat;for;f90"
# Begin Source File

SOURCE=.\anchor.cpp
# End Source File
# Begin Source File

SOURCE=.\audio.cpp
# End Source File
# Begin Source File

SOURCE=.\backgrnd.cpp
# End Source File
# Begin Source File

SOURCE=.\bboard.cpp
# End Source File
# Begin Source File

SOURCE=.\bookmark.cpp
# End Source File
# Begin Source File

SOURCE=.\cppout.cpp
# End Source File
# Begin Source File

SOURCE=.\dllmain.cpp
# End Source File
# Begin Source File

SOURCE=.\EvalCol.cpp
# SUBTRACT CPP /YX /Yc /Yu
# End Source File
# Begin Source File

SOURCE=.\fog.cpp
# End Source File
# Begin Source File

SOURCE=.\inline.cpp
# End Source File
# Begin Source File

SOURCE=.\lod.cpp
# End Source File
# Begin Source File

SOURCE=.\navinfo.cpp
# End Source File
# Begin Source File

SOURCE=.\pmesh.cpp
# End Source File
# Begin Source File

SOURCE=.\polycnt.cpp
# End Source File
# Begin Source File

SOURCE=.\prox.cpp
# End Source File
# Begin Source File

SOURCE=.\sound.cpp
# End Source File
# Begin Source File

SOURCE=.\timer.cpp
# End Source File
# Begin Source File

SOURCE=.\touch.cpp
# End Source File
# Begin Source File

SOURCE=.\vrml2.cpp
# End Source File
# Begin Source File

SOURCE=.\vrml_api.cpp
# End Source File
# Begin Source File

SOURCE=.\vrmlexp.cpp
# End Source File
# Begin Source File

SOURCE=.\vrmlexp.def
# End Source File
# Begin Source File

SOURCE=.\vrmlpch.cpp
# ADD CPP /Yc"vrml.h"
# End Source File
# Begin Source File

SOURCE=..\..\..\lib\bmm.lib
# End Source File
# Begin Source File

SOURCE=..\..\..\lib\core.lib
# End Source File
# Begin Source File

SOURCE=..\..\..\lib\geom.lib
# End Source File
# Begin Source File

SOURCE=..\..\..\lib\gfx.lib
# End Source File
# Begin Source File

SOURCE=..\..\..\lib\mesh.lib
# End Source File
# Begin Source File

SOURCE=..\..\..\lib\maxutil.lib
# End Source File
# Begin Source File

SOURCE=..\..\..\lib\helpsys.lib
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl;fi;fd"
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;cnt;rtf;gif;jpg;jpeg;jpe"
# Begin Source File

SOURCE=.\animcurs.cur
# End Source File
# Begin Source File

SOURCE=.\CROSSHR1.CUR
# End Source File
# Begin Source File

SOURCE=.\dmtlbut.bmp
# End Source File
# Begin Source File

SOURCE=.\dmtlmsk.bmp
# End Source File
# Begin Source File

SOURCE=.\lodcurs.cur
# End Source File
# Begin Source File

SOURCE=.\vrmlexp.rc
# End Source File
# End Group
# End Target
# End Project
