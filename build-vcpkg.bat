REM @echo off

call scripts/vcpkg-settings.bat
call scripts/vcpkg-build-deps.bat
call scripts/vcpkg-build-covise.bat