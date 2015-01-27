; files related to Lesc VR Praktikum

; *********************************
; define your own path here
#define COVISEDIR "C:\COVISE\Autobuild\amdwin64\Covise7.0\covise"
#define DEMODIR "Z:"

; *********************************
; demo files
Source: {#DEMODIR}\Demos\W204\*; DestDir: {app}\Demos\W204; Excludes: .svn\*,*.bak,Thumbs.db,*~;  Components: lesc_vr_praktikum


; *********************************
; configs
Source: {#COVISEDIR}\config\lesc_vr_praktikum\*; DestDir: {app}\covise\config; Components: lesc_vr_praktikum

; *********************************
; environment variable COCONFIG
Source: {#COVISEDIR}\install\lesc_vr_praktikum\common.local.bat; DestDir: {app}\covise; Flags: skipifsourcedoesntexist; Components: lesc_vr_praktikum
