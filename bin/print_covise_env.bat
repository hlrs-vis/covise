@echo off

if exist %COVISEDIR%/winenv.bat (
   if defined ARCHSUFFIX (
       call %COVISEDIR%/winenv.bat %ARCHSUFFIX%
   ) else (
       call %COVISEDIR%/winenv.bat
   )
)

if "%1" equ "" (
    if defined COVISEDIR set COVISEDIR
    if defined COVISE_PATH set COVISE_PATH
    if defined ARCHSUFFIX set ARCHSUFFIX
    if defined COCONFIG set COCONFIG
    if defined VV_SHADER_PATH set VV_SHADER_PATH
    if defined OIV_PSFONT_PATH set OIV_PSFONT_PATH
    if defined PATH set PATH
) else (
    call %COVISEDIR%/scripts/print_envvar.bat %1
)
