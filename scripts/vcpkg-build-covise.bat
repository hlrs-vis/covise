REM @echo off

call vcpkg-settings.bat

set "generator=Visual Studio 16 2019"

set EXTERNLIBS=
set ARCHSUFFIX=vcpkgopt

set cfg=RelWithDebInfo
if "%ARCHSUFFIX%" == "vcpkg" (
    set cfg=Debug
)
set ARCHSUFFIX=vcpkg

set "verb=/consoleloggerparameters:Summary;Verbosity=minimal;ForceNoAlign;ShowTimestamp"
set "par=/m"

mkdir %ARCHSUFFIX%
cd %ARCHSUFFIX%

:COVISE
mkdir build.covise
cd build.covise
cmake -G "%generator%" "-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" -DCOVISE_BUILD_RENDERER:BOOL=OFF -DCOVISE_USE_VISIONARAY:BOOL=OFF ../..
REM cmake -G "%generator%" "-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" -DCOVISE_BUILD_RENDERER:BOOL=OFF ../..
if errorlevel 1 goto UPDIR
msbuild /m covise.sln /p:Configuration=%cfg% %verb% %par%
if errorlevel 1 goto UPDIR

:UPDIR
cd ..
cd ..

:END
