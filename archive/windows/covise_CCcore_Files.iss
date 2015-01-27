; files related to CyberClassroom

; *********************************
; define your own path here
#define VISENSO_BRANCH GetEnv("VISENSO_BRANCH")
#define DEMODIR GetEnv("DEMO_DIR")
#define PROJECTDIR GetEnv("PROJECT_DIR")

[files]

; *********************************
; CyberClassroom core files
Source: {#COVISEDIR}\icons\CyberClassroom.ico; DestDir: {app}; Components: cyberclassroom; Permissions: everyone-full users-full
Source: {#COVISEDIR}\scripts\Setup\install\CC32\OpenCOVERLogo.tif; DestDir: {app}\covise\icons; Permissions: everyone-full users-full
Source: {#VISENSO_BRANCH}\ui\Cyber-Classroom3.0-Install\*; DestDir: {#DBIN}\CyberClassroom\; Flags: skipifsourcedoesntexist recursesubdirs createallsubdirs; Excludes: config.cc.setup.xml,.svn\*,*.bak,Thumbs.db,*~,Kopie von *; Components: cyberclassroom ; Permissions: everyone-full users-full

; *********************************
; demo files
Source: {#DEMODIR}\Demos\Tutorials\TabletTutorial\*; DestDir: {app}\Demos\Tutorials\TabletTutorial\; Excludes: .svn\*,*.bak,Thumbs.db,*~;  Flags: recursesubdirs createallsubdirs; Components: cyberclassroom; Permissions: everyone-full users-full
Source: {#DEMODIR}\Demos\Tutorials\VRiiDTutorial\*; DestDir: {app}\Demos\Tutorials\VRiiDTutorial\; Excludes: .svn\*,*.bak,Thumbs.db,*~;  Flags: recursesubdirs createallsubdirs; Components: cyberclassroom; Permissions: everyone-full users-full
Source: {#DEMODIR}\Demos\About\*; DestDir: {app}\Demos\About\; Excludes: .svn\*,*.bak,Thumbs.db,*~;  Flags: recursesubdirs createallsubdirs; Components: cyberclassroom; Permissions: everyone-full users-full

[dirs]
Name: {app}\Demos\VR4Schule;  Components: cyberclassroom; Permissions: everyone-full users-full
Name: {app}\Demos\CCBerufschulen;  Components: cyberclassroom; Permissions: everyone-full users-full
Name: {app}\Demos\locale;  Components: cyberclassroom; Permissions: everyone-full users-full

[files]
Source: {#DEMODIR}\Demos\Mediathek-Modelle\*; DestDir: {app}\Demos\Mediathek-Modelle; Excludes: airliner_01,.svn\*,*.bak,Thumbs.db,*~; Flags: skipifsourcedoesntexist recursesubdirs createallsubdirs; Components: cyberclassroom; Permissions: everyone-full users-full
Source: {#DEMODIR}\Demos\Mediathek-Modelle\airliner_01; DestDir: {app}\Demos\Mediathek-Modelle\airliner_01; Excludes: .svn\*,*.bak,Thumbs.db,*~; Attribs: hidden; Flags: skipifsourcedoesntexist recursesubdirs createallsubdirs; Components: cyberclassroom; Permissions: everyone-full users-full

; *********************************
; menu locale 
Source: {#DEMODIR}\Demos\locale\*; DestDir: {app}\Demos\locale; Excludes: .svn\*,*.bak,Thumbs.db,*~; Flags: skipifsourcedoesntexist recursesubdirs createallsubdirs; Components: cyberclassroom; Permissions: everyone-full users-full


; *********************************
; batch scripts
Source: {#COVISEDIR}\..\startCyberClassroom.bat; DestDir: {app}; Components: cyberclassroom; Permissions: everyone-full users-full
Source: {#COVISEDIR}\startCOVISE_CCModule.bat; DestDir: {app}; Components: cyberclassroom; Permissions: everyone-full users-full
Source: {#COVISEDIR}\startUNITY_CCModule.bat; DestDir: {app}; Components: cyberclassroom; Permissions: everyone-full users-full


; *********************************
; CC Module Development
Source: {#PROJECTDIR}\Dokumente\Erklaerung_Magnetismus.odp; DestDir: {app}\Modulerstellung\Instruktionsfelder; Components: ccModuleDev; Permissions: everyone-full users-full
Source: {#PROJECTDIR}\Dokumente\pdf2png.bat; DestDir: {app}\Modulerstellung\Instruktionsfelder; Components: ccModuleDev; Permissions: everyone-full users-full
Source: {#PROJECTDIR}\Dokumente\fonts\*; DestDir: {app}\Modulerstellung\fonts; Components: ccModuleDev; Permissions: everyone-full users-full
Source: {#PROJECTDIR}\Dokumente\Fortschrittsbalken\*; DestDir: {app}\Modulerstellung\Instruktionsfelder\Fortschrittsbalken; Components: ccModuleDev; Permissions: everyone-full users-full
Source: {#PROJECTDIR}\Dokumente\icons\*; DestDir: {app}\Modulerstellung\Instruktionsfelder\icons; Components: ccModuleDev; Permissions: everyone-full users-full
Source: {#PROJECTDIR}\Modulerstellung\*; DestDir: {app}\Modulerstellung; Flags: skipifsourcedoesntexist recursesubdirs createallsubdirs; Components: ccModuleDev; Permissions: everyone-full users-full
Source: {#PROJECTDIR}\handbuch\src\LaTeX\Modulhandbuch Beispiel\*; DestDir: {app}\Modulerstellung\Handbuch\Beispiel; Flags: skipifsourcedoesntexist recursesubdirs createallsubdirs; Components: ccModuleDev; Permissions: everyone-full users-full
Source: {#PROJECTDIR}\handbuch\src\LaTeX\Vorlagen\*; DestDir: {app}\Modulerstellung\Handbuch\Vorlagen; Flags: skipifsourcedoesntexist recursesubdirs createallsubdirs; Components: ccModuleDev; Permissions: everyone-full users-full

;***********************************
; unity media player
Source: {#VISENSO_BRANCH}\MediaPlayer\*; DestDir: {app}\covise\bin\MediaPlayer\; Excludes: .svn\*,*.bak,Thumbs.db,*~; Flags: recursesubdirs createallsubdirs; Components: cyberclassroom; Permissions: everyone-full users-full

;***********************************
; Video format switcher for LG TVs 
Source: {#VISENSO_BRANCH}\MediensteuerungLG\x64\Release\SetLG.exe; DestDir: {#DBIN}\MediensteuerungLG; Components: cyberclassroom; Permissions: everyone-full users-full

; *********************************
; configs
Source: {#COVISEDIR}\config\Cyber-Classroom\*; DestDir: {app}\covise\config; Excludes: config.license.xml,config.setup.xml,unityconfig.setup.xml; Components: cyberclassroom; Permissions: everyone-full users-full
Source: {#COVISEDIR}\config\CCBerufschulen\*; DestDir: {app}\covise\config; Components: cyberclassroom; Permissions: everyone-full users-full

; *********************************
; if new installation, use most likely config files (DONT OVERWRITE EXISTING FILES)
Source: {#COVISEDIR}\config\config-license.xml; DestDir: {app}\covise\config; DestName: config.license.xml; Components: cyberclassroom; Permissions: everyone-full users-full; Flags: onlyifdoesntexist
Source: {#COVISEDIR}\config\Cyber-Classroom\config-setup.xml; DestDir: {app}\covise\config; DestName: config.setup.xml; Components: cyberclassroom; Permissions: everyone-full users-full; Flags: onlyifdoesntexist
Source: {#COVISEDIR}\config\Cyber-Classroom\unityconfig-setup.xml; DestDir: {app}\covise\config; DestName: unityconfig.setup.xml; Components: cyberclassroom; Permissions: everyone-full users-full; Flags: onlyifdoesntexist
Source: {#VISENSO_BRANCH}\ui\Cyber-Classroom3.0-Install\bin\config.cc-setup.xml; DestDir: {#DBIN}\CyberClassroom\bin\; DestName: config.cc.setup.xml; Components: cyberclassroom; Permissions: everyone-full users-full; Flags: onlyifdoesntexist
Source: {#COVISEDIR}\startTracking.bat; DestDir: {app}\covise; Components: cyberclassroom; Permissions: everyone-full users-full; Flags: onlyifdoesntexist
