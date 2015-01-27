

#define ARCHSUFFIX GetEnv("COVISE_ARCHSUFFIX")
#if ARCHSUFFIX == ""
  #define ARCHSUFFIX GetEnv("ARCHSUFFIX")
#endif

; get some environment variables
#define DEVELOPMENT GetEnv("COVISE_DEVELOPMENT")
#if DEVELOPMENT == ""
  #define DEVELOPMENT "YES"
#endif

#define TIMEPREFIX GetEnv("COVISE_DISTRO_TIMEPREFIX")
#if TIMEPREFIX != ""
   #define TIMEPREFIX GetDateTimeString('yymmddhhmm/', '_', '');
#endif

#define QT GetEnv("QT_HOME")
#define PNG GetEnv("PNG_HOME")
#define TIFF  GetEnv("TIFF_HOME")
#define JPEG  GetEnv("JPEG_HOME")
#define ZLIB  GetEnv("ZLIB_HOME")
#define XERCES  GetEnv("XERCESC_HOME")
; set the most used pathes
#define GISWALKDIR GetEnv("COVISEDIR")+"\..\common\src\tools\giswalk"
#define EXTERNLIBS GetEnv("EXTERNLIBS")
#define DATADIR "\data\svr"
#define COMMONDIR GetEnv("COVISEDIR") +"\..\common"
#define COVISEDIR GetEnv("COVISEDIR")


#define DIST GISWALKDIR+"\DIST\DIST."+ARCHSUFFIX
#define BIN  COVISEDIR+"\"+ARCHSUFFIX+"\bin"
#define LIB  COVISEDIR+"\"+ARCHSUFFIX+"\lib"
#define DBIN "{app}\giswalk\"+ARCHSUFFIX+"\bin"
#define DDAT "{app}\giswalk\data"
#define DSRC "{app}\giswalk\src"
#define DLIB "{app}\giswalk\"+ARCHSUFFIX+"\lib"


#define DEXT "{app}\giswalk\extern_libs"
#define ICONFILE GISWALKDIR+"\giswalk.ico"


#define SYSTEMROOT GetEnv("SystemRoot")
#if ARCHSUFFIX == "win32opt"
  #define LABEL "_win32opt"
  #define SYS GetEnv("SystemRoot")+"\system32\msvc*??*.dll"
#elif ARCHSUFFIX == "win32"
  #define LABEL "_win32"
  #define SYS GetEnv("SystemRoot")+"\system32\msvc*??*.dll"
#elif ARCHSUFFIX == "vista"
  #define LABEL "_vista"
  #define SYS GetEnv("EXTERNLIBS")+"\runtime\*.exe"
;  #define SYS GetEnv("VCINSTALLDIR")+"\redist\Debug_NonRedist\x86\Microsoft.VC80.DebugCRT\msvc*??*.dll"
#elif ARCHSUFFIX == "vistaopt"
  #define LABEL "_vistaopt"
  #define SYS GetEnv("EXTERNLIBS")+"\runtime\*.exe"
#elif ARCHSUFFIX == "amdwin64"
  #define LABEL "_amdwin64"
  #define SYS GetEnv("EXTERNLIBS")+"\runtime\*.exe"
#elif ARCHSUFFIX == "amdwin64opt"
  #define LABEL "_amdwin64opt"
  #define SYS GetEnv("EXTERNLIBS")+"\runtime\*.exe"
#elif ARCHSUFFIX == "zackel"
  #define LABEL "_zackel"
  #define SYS GetEnv("EXTERNLIBS")+"\runtime\*.exe"
#elif ARCHSUFFIX == "zackelopt"
  #define LABEL "_zackelopt"
  #define SYS GetEnv("EXTERNLIBS")+"\runtime\*.exe"
#elif ARCHSUFFIX == "angus"
  #define LABEL "_angus"
  #define SYS GetEnv("EXTERNLIBS")+"\runtime\*.exe"
#elif ARCHSUFFIX == "angusopt"
  #define LABEL "_angusopt"
  #define SYS GetEnv("EXTERNLIBS")+"\runtime\*.exe"
#else
  #pragma message "Warning: undefined or unknown ARCHSUFFIX! Cannot set SYS variable!"
  #define LABEL "UNKNOWN"
#endif


[Setup]
;compiler-related

PrivilegesRequired=None

#if DEVELOPMENT != "YES"
  #define SUFFIX_DEV "_nodev"
#else
  #define SUFFIX_DEV ""
#endif

OutputDir={#DIST}

#define SUFFIX_VERSION "136"
AppVerName=GisWalk 1.3.6
OutputBaseFilename={#TIMEPREFIX}giswalk_{#SUFFIX_VERSION}{#LABEL}{#SUFFIX_DEV}

;installer-related
#if (ARCHSUFFIX == "amdwin64") || (ARCHSUFFIX == "amdwin64opt") || (ARCHSUFFIX == "angus") || (ARCHSUFFIX == "angusopt")
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
#define SUFFIX_ARCH "x64"
#else
#define SUFFIX_ARCH "x86"
#endif

AppName=GisWalk

AppPublisher=HLRS
AppPublisherURL=http://www.hlrs.de
AppSupportURL=http://www.hlrs.de
AppUpdatesURL=http://www.hlrs.de

ChangesAssociations=true
ChangesEnvironment=true
DefaultDirName={reg:HKLM\SOFTWARE\GisWalk,Path|{pf}\GisWalk}
DefaultGroupName={reg:HKLM\SOFTWARE\GisWalk,StartMenu|GisWalk}
DisableStartupPrompt=true
; cosmetic
SetupIconFile={#ICONFILE}
ShowLanguageDialog=yes

[Types]
Name: standard; Description: GisWalk Standard Installation
Name: devel; Description: GisWalk Developer Installation
[Components]

Name: core; Description: GisWalk core system; Types: standard
Name: source; Description: GisWalk source; Types: standard devel

#if DEVELOPMENT == "YES"
Name: develop; Description: Giswalk development environment; Types: devel

Name: externlibs; Description: Giswalk external libraries (development only)
Name: externlibs/tiff; Description: Tiff development for Windows; Types: devel
Name: externlibs/Xerces; Description: Xerces development for Windows; Types: devel
#endif

[Files]


Source: {#COVISEDIR}\mkspecs\*; DestDir: {app}\giswalk\mkspecs; Components: core

Source: {#BIN}\giswalk.exe; DestDir: {#DBIN}; Components: core

#if ARCHSUFFIX == "vista"
Source: {#EXTERNLIBS}\runtime\vcredist_x86_sp1_secfix.exe; DestDir: {#DLIB}; Flags: recursesubdirs; Components: core
#elif ARCHSUFFIX == "vistaopt"
Source: {#EXTERNLIBS}\runtime\vcredist_x86_sp1_secfix.exe; DestDir: {#DLIB}; Flags: recursesubdirs; Components: core
#elif ARCHSUFFIX == "amdwin64"
Source: {#EXTERNLIBS}\runtime\vcredist_x64_sp1_secfix.exe; DestDir: {#DLIB}; Flags: recursesubdirs; Components: core
#elif ARCHSUFFIX == "amdwin64opt"
Source: {#EXTERNLIBS}\runtime\vcredist_x64_sp1_secfix.exe; DestDir: {#DLIB}; Flags: recursesubdirs; Components: core
#elif ARCHSUFFIX == "zackel"
Source: {#EXTERNLIBS}\runtime\vcredist_x86.exe; DestDir: {#DLIB}; Flags: recursesubdirs; Components: core
#elif ARCHSUFFIX == "zackelopt"
Source: {#EXTERNLIBS}\runtime\vcredist_x86.exe; DestDir: {#DLIB}; Flags: recursesubdirs; Components: core
#elif ARCHSUFFIX == "angus"
Source: {#EXTERNLIBS}\runtime\vcredist_x64.exe; DestDir: {#DLIB}; Flags: recursesubdirs; Components: core
#elif ARCHSUFFIX == "angusopt"
Source: {#EXTERNLIBS}\runtime\vcredist_x64.exe; DestDir: {#DLIB}; Flags: recursesubdirs; Components: core
#endif
Source: {#EXTERNLIBS}\runtime\_version.txt; DestDir: {#DLIB}; Flags: skipifsourcedoesntexist; Components: core



Source: {#XERCES}\lib\*.dll; DestDir: {#DBIN}; Components: core
Source: {#DATADIR}\*; DestDir: {#DDAT}; Excludes: .svn\*; Flags: recursesubdirs; Components: core
Source: {#GISWALKDIR}\*.ico; DestDir: {app}\giswalk; Components: core

Source: {#TIFF}\lib\*.dll; DestDir: {#DBIN}; Components: core
Source: {#EXTERNLIBS}\Xerces\lib\*.dll; DestDir: {#DBIN}; Components: core

Source: {#COVISEDIR}\mkspecs\*.pri; DestDir: {app}\giswalk\mkspecs; Components: develop
Source: {#COMMONDIR}\mkspecs\*.pri; DestDir: {app}\common\mkspecs; Components: develop
Source: {#COMMONDIR}\mkspecs\win32\*; Excludes: .svn\*; DestDir: {app}\common\mkspecs\win32; Components: develop
Source: {#COMMONDIR}\mkspecs\win32opt\*; Excludes: .svn\*; DestDir: {app}\common\mkspecs\win32opt; Components: develop
Source: {#COMMONDIR}\mkspecs\vista\*; Excludes: .svn\*; DestDir: {app}\common\mkspecs\vista; Components: develop
Source: {#COMMONDIR}\mkspecs\vistaopt\*; Excludes: .svn\*; DestDir: {app}\common\mkspecs\vistaopt; Components: develop
Source: {#COMMONDIR}\mkspecs\amdwin64\*; Excludes: .svn\*; DestDir: {app}\common\mkspecs\amdwin64; Components: develop
Source: {#COMMONDIR}\mkspecs\amdwin64opt\*; Excludes: .svn\*; DestDir: {app}\common\mkspecs\amdwin64opt; Components: develop
Source: {#COMMONDIR}\mkspecs\zackel\*; Excludes: .svn\*; DestDir: {app}\common\mkspecs\zackel; Components: develop
Source: {#COMMONDIR}\mkspecs\zackelopt\*; Excludes: .svn\*; DestDir: {app}\common\mkspecs\zackelopt; Components: develop
Source: {#COMMONDIR}\mkspecs\angus\*; Excludes: .svn\*; DestDir: {app}\common\mkspecs\angus; Components: develop
Source: {#COMMONDIR}\mkspecs\angusopt\*; Excludes: .svn\*; DestDir: {app}\common\mkspecs\angusopt; Components: develop
; using Qt version >= 4.5.0 LGPL
Source: {#QT}\bin\qmake.exe; DestDir: {#DEXT}\qt\bin; Components: develop
Source: {#QT}\mkspecs\*; DestDir: {#DEXT}\qt\mkspecs; Components: externlibs; Flags: recursesubdirs

Source: {#GISWALKDIR}\*.h; DestDir: {#DSRC}; Flags: recursesubdirs; Components: develop
Source: {#GISWALKDIR}\*.cpp; DestDir: {#DSRC}; Flags: recursesubdirs; Components: develop

[Registry]


Root: HKCU; Subkey: Environment; ValueType: string; ValueName: ARCHSUFFIX; ValueData: {#ARCHSUFFIX}; Flags: uninsdeletekeyifempty uninsdeletevalue
Root: HKCU; Subkey: Environment; ValueType: string; ValueName: PATH; ValueData: "{app}\GisWalk\{#ARCHSUFFIX}\bin;{app}\GisWalk\{#ARCHSUFFIX}\lib"; Flags: uninsdeletekeyifempty uninsdeletevalue
Root: HKCU; Subkey: Environment; ValueType: string; ValueName: EXTERNLIBS; ValueData: {app}\GisWalk\extern_libs; Flags: uninsdeletekeyifempty uninsdeletevalue
Root: HKLM; Subkey: SYSTEM\CurrentControlSet\Control\Session Manager\Environment; ValueType: string; ValueName: EXTERNLIBS; ValueData: {app}\GisWalk\extern_libs; Flags: uninsdeletekeyifempty uninsdeletevalue
Root: HKLM; Subkey: SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\GisWalk.exe; ValueType: string; ValueData: {app}\GisWalk\{#ARCHSUFFIX}\bin\giswalk.exe; Flags: uninsdeletekeyifempty uninsdeletevalue



[UninstallDelete]


[Tasks]
Name: desktopicon; Description: Icons on &Desktop; GroupDescription: Desctop Icons:
Name: startupcion; Description: Icons into &Startup; GroupDescription: Startup Icons:; Flags: unchecked


[Icons]

Name: {group}\GisWalk; Filename: {app}\giswalk\{#ARCHSUFFIX}\bin\giswalk.exe; Comment: Start GisWalk; IconFilename: {app}\giswalk\giswalk.ico; Flags: createonlyiffileexists dontcloseonexit
Name: {commondesktop}\GisWalk; Filename: {app}\giswalk\{#ARCHSUFFIX}\bin\giswalk.exe; Comment: Start GisWalk; IconFilename: {app}\giswalk\giswalk.ico; Tasks: desktopicon; Flags: createonlyiffileexists dontcloseonexit
Name: {group}\Uninstall GisWalk; Filename: {uninstallexe}


[Run]

#if ARCHSUFFIX == "vista"
Filename: {app}\giswalk\{#ARCHSUFFIX}\lib\vcredist_x86_sp1_secfix.exe; Parameters: /Q; Description: Install VisualStudio 2005 SP1 Runtime (incl. ATL sec.fix); Flags: postinstall
#elif ARCHSUFFIX == "vistaopt"
Filename: {app}\giswalk\{#ARCHSUFFIX}\lib\vcredist_x86_sp1_secfix.exe; Parameters: /Q; Description: Install VisualStudio 2005 SP1 Runtime (incl. ATL sec.fix); Flags: postinstall
#elif ARCHSUFFIX == "amdwin64"
Filename: {app}\giswalk\{#ARCHSUFFIX}\lib\vcredist_x64_sp1_secfix.exe; Parameters: /Q; Description: Install VisualStudio 2005 SP1 Runtime (incl. ATL sec.fix); Flags: postinstall
#elif ARCHSUFFIX == "amdwin64opt"
Filename: {app}\giswalk\{#ARCHSUFFIX}\lib\vcredist_x64_sp1_secfix.exe; Parameters: /Q; Description: Install VisualStudio 2005 SP1 Runtime (incl. ATL sec.fix); Flags: postinstall
#elif ARCHSUFFIX == "angus"
Filename: {app}\giswalk\{#ARCHSUFFIX}\lib\vcredist_x64.exe; Parameters: /Q; Description: Install VisualStudio 2008 Runtime; Flags: postinstall
#elif ARCHSUFFIX == "angusopt"
Filename: {app}\giswalk\{#ARCHSUFFIX}\lib\vcredist_x64.exe; Parameters: /Q; Description: Install VisualStudio 2008 Runtime; Flags: postinstall
#elif ARCHSUFFIX == "zackel"
Filename: {app}\giswalk\{#ARCHSUFFIX}\lib\vcredist_x86.exe; Parameters: /Q; Description: Install VisualStudio 2008 Runtime; Flags: postinstall
#elif ARCHSUFFIX == "zackelopt"
Filename: {app}\giswalk\{#ARCHSUFFIX}\lib\vcredist_x86.exe; Parameters: /Q; Description: Install VisualStudio 2008 Runtime; Flags: postinstall
#endif
