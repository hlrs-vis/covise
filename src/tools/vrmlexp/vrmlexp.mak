# Microsoft Developer Studio Generated NMAKE File, Based on vrmlexp.dsp
!IF "$(CFG)" == ""
CFG=vrmlexp - Win32 Release
!MESSAGE No configuration specified. Defaulting to vrmlexp - Win32 Release.
!ENDIF 

!IF "$(CFG)" != "vrmlexp - Win32 Release" && "$(CFG)" != "vrmlexp - Win32 Debug" && "$(CFG)" != "vrmlexp - Win32 Hybrid"
!MESSAGE Invalid configuration "$(CFG)" specified.
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
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "vrmlexp - Win32 Release"

OUTDIR=.\Release
INTDIR=.\Release

ALL : "..\..\..\..\maxsdk\plugin\vrmlexp.dle"


CLEAN :
	-@erase "$(INTDIR)\anchor.obj"
	-@erase "$(INTDIR)\audio.obj"
	-@erase "$(INTDIR)\backgrnd.obj"
	-@erase "$(INTDIR)\bboard.obj"
	-@erase "$(INTDIR)\bookmark.obj"
	-@erase "$(INTDIR)\cppout.obj"
	-@erase "$(INTDIR)\dllmain.obj"
	-@erase "$(INTDIR)\EvalCol.obj"
	-@erase "$(INTDIR)\fog.obj"
	-@erase "$(INTDIR)\inline.obj"
	-@erase "$(INTDIR)\lod.obj"
	-@erase "$(INTDIR)\navinfo.obj"
	-@erase "$(INTDIR)\pmesh.obj"
	-@erase "$(INTDIR)\polycnt.obj"
	-@erase "$(INTDIR)\prox.obj"
	-@erase "$(INTDIR)\sound.obj"
	-@erase "$(INTDIR)\timer.obj"
	-@erase "$(INTDIR)\touch.obj"
	-@erase "$(INTDIR)\vrml2.obj"
	-@erase "$(INTDIR)\vrml_api.obj"
	-@erase "$(INTDIR)\vrmlexp.obj"
	-@erase "$(INTDIR)\vrmlexp.pch"
	-@erase "$(INTDIR)\vrmlexp.res"
	-@erase "$(INTDIR)\vrmlpch.obj"
	-@erase "$(OUTDIR)\vrmlexp.exp"
	-@erase "$(OUTDIR)\vrmlexp.lib"
	-@erase "..\..\..\..\maxsdk\plugin\vrmlexp.dle"
	-@erase ".\vrmlexp.idb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP_PROJ=/nologo /G6 /MD /W3 /GX /O2 /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "NDEBUG" /D "_LEC_" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\vrmlexp.pch" /Yu"vrml.h" /Fo"$(INTDIR)\\" /Fd"vrmlexp.pdb" /FD /c 
MTL_PROJ=/nologo /D "NDEBUG" /mktyplib203 /win32 
RSC_PROJ=/l 0x409 /fo"$(INTDIR)\vrmlexp.res" /i "..\..\..\include" /d "NDEBUG" 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\vrmlexp.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib version.lib ..\..\..\lib\maxscrpt.lib /nologo /base:"0x641d0000" /subsystem:windows /dll /incremental:no /pdb:"$(OUTDIR)\vrmlexp.pdb" /machine:I386 /def:".\vrmlexp.def" /out:"..\..\..\..\maxsdk\plugin\vrmlexp.dle" /implib:"$(OUTDIR)\vrmlexp.lib" /release 
DEF_FILE= \
	".\vrmlexp.def"
LINK32_OBJS= \
	"$(INTDIR)\anchor.obj" \
	"$(INTDIR)\audio.obj" \
	"$(INTDIR)\backgrnd.obj" \
	"$(INTDIR)\bboard.obj" \
	"$(INTDIR)\bookmark.obj" \
	"$(INTDIR)\cppout.obj" \
	"$(INTDIR)\dllmain.obj" \
	"$(INTDIR)\EvalCol.obj" \
	"$(INTDIR)\fog.obj" \
	"$(INTDIR)\inline.obj" \
	"$(INTDIR)\lod.obj" \
	"$(INTDIR)\navinfo.obj" \
	"$(INTDIR)\pmesh.obj" \
	"$(INTDIR)\polycnt.obj" \
	"$(INTDIR)\prox.obj" \
	"$(INTDIR)\sound.obj" \
	"$(INTDIR)\timer.obj" \
	"$(INTDIR)\touch.obj" \
	"$(INTDIR)\vrml2.obj" \
	"$(INTDIR)\vrml_api.obj" \
	"$(INTDIR)\vrmlexp.obj" \
	"$(INTDIR)\vrmlpch.obj" \
	"$(INTDIR)\vrmlexp.res" \
	"..\..\..\lib\bmm.lib" \
	"..\..\..\lib\core.lib" \
	"..\..\..\lib\geom.lib" \
	"..\..\..\lib\gfx.lib" \
	"..\..\..\lib\mesh.lib" \
	"..\..\..\lib\maxutil.lib" \
	"..\..\..\lib\helpsys.lib"

"..\..\..\..\maxsdk\plugin\vrmlexp.dle" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ELSEIF  "$(CFG)" == "vrmlexp - Win32 Debug"

OUTDIR=.\Debug
INTDIR=.\Debug

ALL : "..\..\..\..\maxsdk\plugin\vrmlexp.dle"


CLEAN :
	-@erase "$(INTDIR)\anchor.obj"
	-@erase "$(INTDIR)\audio.obj"
	-@erase "$(INTDIR)\backgrnd.obj"
	-@erase "$(INTDIR)\bboard.obj"
	-@erase "$(INTDIR)\bookmark.obj"
	-@erase "$(INTDIR)\cppout.obj"
	-@erase "$(INTDIR)\dllmain.obj"
	-@erase "$(INTDIR)\EvalCol.obj"
	-@erase "$(INTDIR)\fog.obj"
	-@erase "$(INTDIR)\inline.obj"
	-@erase "$(INTDIR)\lod.obj"
	-@erase "$(INTDIR)\navinfo.obj"
	-@erase "$(INTDIR)\pmesh.obj"
	-@erase "$(INTDIR)\polycnt.obj"
	-@erase "$(INTDIR)\prox.obj"
	-@erase "$(INTDIR)\sound.obj"
	-@erase "$(INTDIR)\timer.obj"
	-@erase "$(INTDIR)\touch.obj"
	-@erase "$(INTDIR)\vrml2.obj"
	-@erase "$(INTDIR)\vrml_api.obj"
	-@erase "$(INTDIR)\vrmlexp.obj"
	-@erase "$(INTDIR)\vrmlexp.pch"
	-@erase "$(INTDIR)\vrmlexp.res"
	-@erase "$(INTDIR)\vrmlpch.obj"
	-@erase "$(OUTDIR)\vrmlexp.exp"
	-@erase "$(OUTDIR)\vrmlexp.lib"
	-@erase "$(OUTDIR)\vrmlexp.pdb"
	-@erase "..\..\..\..\maxsdk\plugin\vrmlexp.dle"
	-@erase "..\..\..\..\maxsdk\plugin\vrmlexp.ilk"
	-@erase ".\vrmlexp.idb"
	-@erase ".\vrmlexp.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP_PROJ=/nologo /G6 /MDd /W3 /Gm /GX /ZI /Od /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_LEC_" /Fp"$(INTDIR)\vrmlexp.pch" /Yu"vrml.h" /Fo"$(INTDIR)\\" /Fd"vrmlexp.pdb" /FD /c 
MTL_PROJ=/nologo /D "_DEBUG" /mktyplib203 /win32 
RSC_PROJ=/l 0x409 /fo"$(INTDIR)\vrmlexp.res" /i "..\..\..\include" /d "_DEBUG" 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\vrmlexp.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib version.lib ..\..\..\lib\maxscrpt.lib /nologo /base:"0x641d0000" /subsystem:windows /dll /incremental:yes /pdb:"$(OUTDIR)\vrmlexp.pdb" /debug /machine:I386 /def:".\vrmlexp.def" /out:"..\..\..\..\maxsdk\plugin\vrmlexp.dle" /implib:"$(OUTDIR)\vrmlexp.lib" 
DEF_FILE= \
	".\vrmlexp.def"
LINK32_OBJS= \
	"$(INTDIR)\anchor.obj" \
	"$(INTDIR)\audio.obj" \
	"$(INTDIR)\backgrnd.obj" \
	"$(INTDIR)\bboard.obj" \
	"$(INTDIR)\bookmark.obj" \
	"$(INTDIR)\cppout.obj" \
	"$(INTDIR)\dllmain.obj" \
	"$(INTDIR)\EvalCol.obj" \
	"$(INTDIR)\fog.obj" \
	"$(INTDIR)\inline.obj" \
	"$(INTDIR)\lod.obj" \
	"$(INTDIR)\navinfo.obj" \
	"$(INTDIR)\pmesh.obj" \
	"$(INTDIR)\polycnt.obj" \
	"$(INTDIR)\prox.obj" \
	"$(INTDIR)\sound.obj" \
	"$(INTDIR)\timer.obj" \
	"$(INTDIR)\touch.obj" \
	"$(INTDIR)\vrml2.obj" \
	"$(INTDIR)\vrml_api.obj" \
	"$(INTDIR)\vrmlexp.obj" \
	"$(INTDIR)\vrmlpch.obj" \
	"$(INTDIR)\vrmlexp.res" \
	"..\..\..\lib\bmm.lib" \
	"..\..\..\lib\core.lib" \
	"..\..\..\lib\geom.lib" \
	"..\..\..\lib\gfx.lib" \
	"..\..\..\lib\mesh.lib" \
	"..\..\..\lib\maxutil.lib" \
	"..\..\..\lib\helpsys.lib"

"..\..\..\..\maxsdk\plugin\vrmlexp.dle" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ELSEIF  "$(CFG)" == "vrmlexp - Win32 Hybrid"

OUTDIR=.\Hybrid
INTDIR=.\Hybrid

ALL : "..\..\..\..\maxsdk\plugin\vrmlexp.dle"


CLEAN :
	-@erase "$(INTDIR)\anchor.obj"
	-@erase "$(INTDIR)\audio.obj"
	-@erase "$(INTDIR)\backgrnd.obj"
	-@erase "$(INTDIR)\bboard.obj"
	-@erase "$(INTDIR)\bookmark.obj"
	-@erase "$(INTDIR)\cppout.obj"
	-@erase "$(INTDIR)\dllmain.obj"
	-@erase "$(INTDIR)\EvalCol.obj"
	-@erase "$(INTDIR)\fog.obj"
	-@erase "$(INTDIR)\inline.obj"
	-@erase "$(INTDIR)\lod.obj"
	-@erase "$(INTDIR)\navinfo.obj"
	-@erase "$(INTDIR)\pmesh.obj"
	-@erase "$(INTDIR)\polycnt.obj"
	-@erase "$(INTDIR)\prox.obj"
	-@erase "$(INTDIR)\sound.obj"
	-@erase "$(INTDIR)\timer.obj"
	-@erase "$(INTDIR)\touch.obj"
	-@erase "$(INTDIR)\vrml2.obj"
	-@erase "$(INTDIR)\vrml_api.obj"
	-@erase "$(INTDIR)\vrmlexp.obj"
	-@erase "$(INTDIR)\vrmlexp.pch"
	-@erase "$(INTDIR)\vrmlexp.res"
	-@erase "$(INTDIR)\vrmlpch.obj"
	-@erase "$(OUTDIR)\vrmlexp.exp"
	-@erase "$(OUTDIR)\vrmlexp.lib"
	-@erase "$(OUTDIR)\vrmlexp.pdb"
	-@erase "..\..\..\..\maxsdk\plugin\vrmlexp.dle"
	-@erase "..\..\..\..\maxsdk\plugin\vrmlexp.ilk"
	-@erase ".\vrmlexp.idb"
	-@erase ".\vrmlexp.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP_PROJ=/nologo /G6 /MD /W3 /Gm /GX /ZI /Od /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_LEC_" /Fp"$(INTDIR)\vrmlexp.pch" /Yu"vrml.h" /Fo"$(INTDIR)\\" /Fd"vrmlexp.pdb" /FD /c 
MTL_PROJ=/nologo /D "_DEBUG" /mktyplib203 /win32 
RSC_PROJ=/l 0x409 /fo"$(INTDIR)\vrmlexp.res" /i "..\..\..\include" /d "_DEBUG" 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\vrmlexp.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib version.lib ..\..\..\lib\maxscrpt.lib /nologo /base:"0x641d0000" /subsystem:windows /dll /incremental:yes /pdb:"$(OUTDIR)\vrmlexp.pdb" /debug /machine:I386 /def:".\vrmlexp.def" /out:"..\..\..\..\maxsdk\plugin\vrmlexp.dle" /implib:"$(OUTDIR)\vrmlexp.lib" 
DEF_FILE= \
	".\vrmlexp.def"
LINK32_OBJS= \
	"$(INTDIR)\anchor.obj" \
	"$(INTDIR)\audio.obj" \
	"$(INTDIR)\backgrnd.obj" \
	"$(INTDIR)\bboard.obj" \
	"$(INTDIR)\bookmark.obj" \
	"$(INTDIR)\cppout.obj" \
	"$(INTDIR)\dllmain.obj" \
	"$(INTDIR)\EvalCol.obj" \
	"$(INTDIR)\fog.obj" \
	"$(INTDIR)\inline.obj" \
	"$(INTDIR)\lod.obj" \
	"$(INTDIR)\navinfo.obj" \
	"$(INTDIR)\pmesh.obj" \
	"$(INTDIR)\polycnt.obj" \
	"$(INTDIR)\prox.obj" \
	"$(INTDIR)\sound.obj" \
	"$(INTDIR)\timer.obj" \
	"$(INTDIR)\touch.obj" \
	"$(INTDIR)\vrml2.obj" \
	"$(INTDIR)\vrml_api.obj" \
	"$(INTDIR)\vrmlexp.obj" \
	"$(INTDIR)\vrmlpch.obj" \
	"$(INTDIR)\vrmlexp.res" \
	"..\..\..\lib\bmm.lib" \
	"..\..\..\lib\core.lib" \
	"..\..\..\lib\geom.lib" \
	"..\..\..\lib\gfx.lib" \
	"..\..\..\lib\mesh.lib" \
	"..\..\..\lib\maxutil.lib" \
	"..\..\..\lib\helpsys.lib"

"..\..\..\..\maxsdk\plugin\vrmlexp.dle" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("vrmlexp.dep")
!INCLUDE "vrmlexp.dep"
!ELSE 
!MESSAGE Warning: cannot find "vrmlexp.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "vrmlexp - Win32 Release" || "$(CFG)" == "vrmlexp - Win32 Debug" || "$(CFG)" == "vrmlexp - Win32 Hybrid"
SOURCE=.\anchor.cpp

"$(INTDIR)\anchor.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\audio.cpp

"$(INTDIR)\audio.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\backgrnd.cpp

"$(INTDIR)\backgrnd.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\bboard.cpp

"$(INTDIR)\bboard.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\bookmark.cpp

"$(INTDIR)\bookmark.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\cppout.cpp

"$(INTDIR)\cppout.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\dllmain.cpp

"$(INTDIR)\dllmain.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\EvalCol.cpp

!IF  "$(CFG)" == "vrmlexp - Win32 Release"

CPP_SWITCHES=/nologo /G6 /MD /W3 /GX /O2 /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "NDEBUG" /D "_LEC_" /D "WIN32" /D "_WINDOWS" /Fo"$(INTDIR)\\" /Fd"vrmlexp.pdb" /FD /c 

"$(INTDIR)\EvalCol.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "vrmlexp - Win32 Debug"

CPP_SWITCHES=/nologo /G6 /MDd /W3 /Gm /GX /ZI /Od /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_LEC_" /Fo"$(INTDIR)\\" /Fd"vrmlexp.pdb" /FD /c 

"$(INTDIR)\EvalCol.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "vrmlexp - Win32 Hybrid"

CPP_SWITCHES=/nologo /G6 /MD /W3 /Gm /GX /ZI /Od /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_LEC_" /Fo"$(INTDIR)\\" /Fd"vrmlexp.pdb" /FD /c 

"$(INTDIR)\EvalCol.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\fog.cpp

"$(INTDIR)\fog.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\inline.cpp

"$(INTDIR)\inline.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\lod.cpp

"$(INTDIR)\lod.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\navinfo.cpp

"$(INTDIR)\navinfo.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\pmesh.cpp

"$(INTDIR)\pmesh.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\polycnt.cpp

"$(INTDIR)\polycnt.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\prox.cpp

"$(INTDIR)\prox.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\sound.cpp

"$(INTDIR)\sound.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\timer.cpp

"$(INTDIR)\timer.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\touch.cpp

"$(INTDIR)\touch.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\vrml2.cpp

"$(INTDIR)\vrml2.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\vrml_api.cpp

"$(INTDIR)\vrml_api.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\vrmlexp.cpp

"$(INTDIR)\vrmlexp.obj" : $(SOURCE) "$(INTDIR)" "$(INTDIR)\vrmlexp.pch"


SOURCE=.\vrmlpch.cpp

!IF  "$(CFG)" == "vrmlexp - Win32 Release"

CPP_SWITCHES=/nologo /G6 /MD /W3 /GX /O2 /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "NDEBUG" /D "_LEC_" /D "WIN32" /D "_WINDOWS" /Fp"$(INTDIR)\vrmlexp.pch" /Yc"vrml.h" /Fo"$(INTDIR)\\" /Fd"vrmlexp.pdb" /FD /c 

"$(INTDIR)\vrmlpch.obj"	"$(INTDIR)\vrmlexp.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "vrmlexp - Win32 Debug"

CPP_SWITCHES=/nologo /G6 /MDd /W3 /Gm /GX /ZI /Od /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_LEC_" /Fp"$(INTDIR)\vrmlexp.pch" /Yc"vrml.h" /Fo"$(INTDIR)\\" /Fd"vrmlexp.pdb" /FD /c 

"$(INTDIR)\vrmlpch.obj"	"$(INTDIR)\vrmlexp.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "vrmlexp - Win32 Hybrid"

CPP_SWITCHES=/nologo /G6 /MD /W3 /Gm /GX /ZI /Od /I "..\..\..\include" /I "..\..\..\include\maxscrpt" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_LEC_" /Fp"$(INTDIR)\vrmlexp.pch" /Yc"vrml.h" /Fo"$(INTDIR)\\" /Fd"vrmlexp.pdb" /FD /c 

"$(INTDIR)\vrmlpch.obj"	"$(INTDIR)\vrmlexp.pch" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\vrmlexp.rc

"$(INTDIR)\vrmlexp.res" : $(SOURCE) "$(INTDIR)"
	$(RSC) $(RSC_PROJ) $(SOURCE)



!ENDIF 

