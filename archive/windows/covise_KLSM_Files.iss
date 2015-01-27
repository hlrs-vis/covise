; files related to KLSM Praktikum

; *********************************
; define your own path here
#define COVISEDIR "E:\workspace\Covise7.0\covise"
#define RESOURCEDIR "\\nas\data\Kunden\klsmartin\resources"
#define SPLASHDIR "E:\workspace\Visenso\SplashClient"

; *********************************
; ressource files
Source: {#RESOURCEDIR}\*; DestDir: {app}\KLSM; Flags: recursesubdirs;  Excludes: .svn\*,*.bak,Thumbs.db,*~;  Components: KLSM

; *********************************
; splash screen
Source: {#SPLASHDIR}\x64\Release\marWORLD3D.exe; DestDir: {app}\covise; Components: KLSM
Source: {#SPLASHDIR}\Splash.bmp; DestDir: {app}\covise; Components: KLSM

; *********************************
; config files
Source: {#COVISEDIR}\config\KLSM\*; DestDir: {app}\covise\config; Components: KLSM

; *********************************
; getting started
Source: {#COVISEDIR}\Python\bin\vr-prepare\documents\GettingStartedKLSM\*.*; DestDir: {app}\covise\Python\bin\vr-prepare\documents\GettingStartedKLSM; Flags: recursesubdirs; Components: KLSM

