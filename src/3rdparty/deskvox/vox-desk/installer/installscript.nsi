; installscript.nsi
;
; This script creates an installer for DeskVOX which can be compiled with the
; Nullsoft Install System V2.0.

!define VER_MAJOR 2
!define VER_MINOR 01b

; The name of the installer:
Name "DeskVOX"

Caption "DeskVOX - Setup"

; The self-extracting installer file to write:
OutFile "DeskVOX${VER_MAJOR}_${VER_MINOR}.exe"

; The icon to use for the installer:
Icon deskvox.ico

; Images for section selection:
;EnabledBitmap enabled.bmp
;DisabledBitmap disabled.bmp

; The default installation directory
InstallDir $PROGRAMFILES\DeskVOX

; Registry key to check for directory (so if you install again, it will 
; overwrite the old one automatically)
InstallDirRegKey HKLM SOFTWARE\DeskVOX "Install_Dir"

; The text to prompt the user to enter a directory
ComponentText "This will install DeskVOX on your computer."

; The text to prompt the user to enter a directory
DirText "Choose a directory to install in to:"

; The stuff to install
Section "DeskVOX (required)"

  ; Set output path to the installation directory.
  SetOutPath $INSTDIR

  ; Files to install:
  File ..\bin\win\DeskVOX.exe
  File ..\bin\win\vconv.exe
  File ..\bin\win\cg.dll
  File ..\bin\win\cgGL.dll
  File ..\bin\win\glut32.dll
  File ..\readme.txt
  File ..\license.txt
  File ..\..\virvo\shader\vv_shader01.cg
  File ..\..\virvo\shader\vv_shader02.cg
  File ..\..\virvo\shader\vv_shader03.cg
  File ..\..\virvo\shader\vv_shader04.cg
  File ..\..\virvo\shader\vv_shader05.cg
  File ..\..\virvo\shader\vv_shader06.cg
  File ..\..\virvo\shader\vv_shader07.cg
  File ..\..\virvo\shader\vv_shader08.cg
  File ..\..\virvo\shader\vv_shader09.cg
  File ..\..\virvo\shader\vv_shader10.cg

  ; Set output path to the installation directory.
  SetOutPath $INSTDIR\examples
  
  ; Examples to install:
  File ..\examples\checkercube.rvf
  File ..\examples\softpyramid.xvf
  File ..\examples\moviescript.vms

  ; Write the installation path into the registry
  WriteRegStr HKLM SOFTWARE\DeskVOX "Install_Dir" "$INSTDIR"

  ; Write the uninstall keys for Windows
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DeskVOX" "DisplayName" "DeskVOX (remove only)"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DeskVOX" "UninstallString" '"$INSTDIR\uninstall.exe"'
  
  ; Delete any previously created uninstaller:
  Delete $INSTDIR\uninstall.exe 

  ; Create uninstaller:
  WriteUninstaller $INSTDIR\uninstall.exe
SectionEnd

; optional section
Section "Start Menu Shortcuts (recommended)"
  CreateDirectory "$SMPROGRAMS\DeskVOX"
  CreateShortCut "$SMPROGRAMS\DeskVOX\Uninstall.lnk" "$INSTDIR\uninstall.exe" "" "$INSTDIR\uninstall.exe" 0
  CreateShortCut "$SMPROGRAMS\DeskVOX\DeskVOX.lnk" "$INSTDIR\DeskVOX.exe" "" "$INSTDIR\DeskVOX.exe" 0
SectionEnd

; Create file association in registry:
Section "Associate File Extension .xvf (recommended)"
    ; back up old value of .xvf
  !define Index "Line${__LINE__}"
    ReadRegStr $1 HKCR ".xvf" ""
    StrCmp $1 "" "${Index}-NoBackup"
      StrCmp $1 "VirvoVolume" "${Index}-NoBackup"
      WriteRegStr HKCR ".xvf" "backup_val" $1
  "${Index}-NoBackup:"
    WriteRegStr HKCR ".xvf" "" "VirvoVolume"
    ReadRegStr $0 HKCR "VirvoVolume" ""
    StrCmp $0 "" 0 "${Index}-Skip"
    WriteRegStr HKCR "VirvoVolume" "" "DeskVOX Volume File"
    WriteRegStr HKCR "VirvoVolume\shell" "" "open"
    WriteRegStr HKCR "VirvoVolume\DefaultIcon" "" "$INSTDIR\execute.exe,0"
  "${Index}-Skip:"
    WriteRegStr HKCR "VirvoVolume\shell\open\command" "" \
      '$INSTDIR\execute.exe "%1"'
    WriteRegStr HKCR "VirvoVolume\shell\edit" "" "Show Volume File"
    WriteRegStr HKCR "VirvoVolume\shell\edit\command" "" \
      '$INSTDIR\execute.exe "%1"'
  !undef Index

  ; Restore script:
  !define Index "Line${__LINE__}"
    ReadRegStr $1 HKCR ".xvf" ""
    StrCmp $1 "VirvoVolume" 0 "${Index}-NoOwn" ; only do this if we own it
      ReadRegStr $1 HKCR ".xvf" "backup_val"
      StrCmp $1 "" 0 "${Index}-Restore" ; if backup="" then delete the whole key
        DeleteRegKey HKCR ".xvf"
      Goto "${Index}-NoOwn"
  "${Index}-Restore:"
        WriteRegStr HKCR ".xvf" "" $1
        DeleteRegValue HKCR ".xvf" "backup_val"
     
      DeleteRegKey HKCR "VirvoVolume" ;Delete key with association settings

  "${Index}-NoOwn:"
  !undef Index
SectionEnd

; uninstall stuff

UninstallText "This will uninstall DeskVOX. Hit next to continue."

; special uninstall section.
Section "Uninstall"
  ; remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DeskVOX"
  DeleteRegKey HKLM SOFTWARE\DeskVOX

  ; remove files
  Delete $INSTDIR\DeskVOX.exe
  Delete $INSTDIR\vconv.exe
  Delete $INSTDIR\cg.dll
  Delete $INSTDIR\cgGL.dll
  Delete $INSTDIR\glut32.dll
  Delete $INSTDIR\readme.txt
  Delete $INSTDIR\license.txt
  Delete $INSTDIR\vv_shader01.cg
  Delete $INSTDIR\vv_shader02.cg
  Delete $INSTDIR\vv_shader03.cg
  Delete $INSTDIR\vv_shader04.cg
  Delete $INSTDIR\vv_shader05.cg
  Delete $INSTDIR\vv_shader06.cg
  Delete $INSTDIR\vv_shader07.cg
  Delete $INSTDIR\vv_shader08.cg
  Delete $INSTDIR\vv_shader09.cg
  Delete $INSTDIR\vv_shader10.cg
  Delete $INSTDIR\examples\checkercube.rvf
  Delete $INSTDIR\examples\softpyramid.xvf
  Delete $INSTDIR\examples\moviescript.vms

  ; MUST REMOVE UNINSTALLER, too
  Delete $INSTDIR\uninstall.exe

  ; remove shortcuts, if any.
  Delete "$SMPROGRAMS\DeskVOX\*.*"

  SetOutPath $SMPROGRAMS

  ; remove directories used.
  RMDir "$SMPROGRAMS\DeskVOX\examples"
  RMDir "$SMPROGRAMS\DeskVOX"
SectionEnd

; eof
