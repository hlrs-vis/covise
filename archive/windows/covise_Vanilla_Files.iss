; files related to a vanilla distro

; *********************************
; configs
Source: {#COVISEDIR}\config\config*.xml; DestDir: {app}\covise\config; Excludes: config.license.xml,config-license.xml; Components: core; Permissions: everyone-full users-full
Source: {#COVISEDIR}\config\config-license.xml; DestDir: {app}\covise\config; DestName: config.license.xml; Components: core; Permissions: everyone-full users-full; Flags: onlyifdoesntexist

