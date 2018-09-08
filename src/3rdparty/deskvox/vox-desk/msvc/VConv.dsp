# Microsoft Developer Studio Project File - Name="VConv" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=VConv - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "VConv.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "VConv.mak" CFG="VConv - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "VConv - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe
# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir ".\VConv_Debug"
# PROP BASE Intermediate_Dir ".\VConv_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir ".\VConv_Debug"
# PROP Intermediate_Dir ".\VConv_Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
MTL=midl.exe
# ADD BASE MTL /nologo /tlb".\Debug\VConv.tlb" /win32
# ADD MTL /nologo /tlb".\Debug\VConv.tlb" /win32
# ADD BASE CPP /nologo /W4 /GX /ZI /Od /I "..\src" /I "..\include" /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_STANDARD_C_PLUS_PLUS" /D "NODLL" /D "_MBCS" /FR /Fp"VConv_Debug/VConv.pch" /YX /Fo"VConv_Debug/" /Fd"VConv_Debug/" /GZ /c
# ADD CPP /nologo /MTd /W4 /GR /GX /ZI /Od /I "..\include" /I "..\..\virvo" /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /FR /Fo"VConv_Debug/" /Fd"VConv_Debug/" /GZ /Zm800 /c
# ADD BASE RSC /l 0x407 /d "_DEBUG"
# ADD RSC /l 0x407 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /pdb:"VConv_Debug\vconv.pdb" /debug /machine:I386 /out:"..\bin\win32\vconv.exe" /pdbtype:sept
# SUBTRACT BASE LINK32 /pdb:none
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib odbc32.lib odbccp32.lib libvirvo.lib /nologo /subsystem:console /pdb:"VConv_Debug\vconv.pdb" /debug /machine:I386 /nodefaultlib:"libcd.lib" /out:"..\bin\win\vconv.exe" /pdbtype:sept /libpath:"..\..\virvo\obj"
# SUBTRACT LINK32 /pdb:none /nodefaultlib
# Begin Target

# Name "VConv - Win32 Debug"
# Begin Group "Source-Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\src\vvconv.cpp
DEP_CPP_VVCON=\
	"..\..\virvo\vvarray.h"\
	"..\..\virvo\vvdebugmsg.h"\
	"..\..\virvo\vvexport.h"\
	"..\..\virvo\vvfileio.h"\
	"..\..\virvo\vvsllist.h"\
	"..\..\virvo\vvtfwidget.h"\
	"..\..\virvo\vvtoolshed.h"\
	"..\..\virvo\vvtransfunc.h"\
	"..\..\virvo\vvvecmath.h"\
	"..\..\virvo\vvvirvo.h"\
	"..\..\virvo\vvvoldesc.h"\
	"..\src\vvconv.h"\
	
# End Source File
# End Group
# Begin Group "Header-Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\src\vvconv.h
# End Source File
# End Group
# Begin Group "Ressourcendateien"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# End Target
# End Project
