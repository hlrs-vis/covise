REM @echo off

if "%VCPKG_ROOT%" EQU "" (
    set VCPKG_ROOT=C:\vcpkg
)

if NOT EXIST "%VCPKG_ROOT%\.vcpkg-root" (
    VCPKG_ROOT has to be set to the root directory of your vcpkg installation
    exit 1
    goto END
)

set VCPKG_DEFAULT_TRIPLET=x64-windows

set "generator=Visual Studio 16 2019"

set EXTERNLIBS=
set ARCHSUFFIX=vcpkgopt
set "CURDIR=%CD%"

:TESTUP
if NOT EXIST .covise.sh (
   cd ..
   goto TESTUP
)

set "COVISEDIR=%CD%"
set "COVISEDESTDIR=%COVISEDIR%"

cd "%CURDIR%"