REM @args: covise dir, archsuffix, dependencyPath, generator, overwrite
set EXTERNLIBS=%3
set VCPKG_ROOT=%3

set COVISEDIR=%1
call %1/winenv.bat zebu
Rem write down covise environment to reuse in vs code debugger settings and CMake wrapper
set > %1/.vscode/configureCovise/covise.env
cd %1/.vscode/configureCovise
rmdir /s /q build
mkdir build
cd build
set GENERATOR=%4
cmake -DCMAKE_PREFIX_PATH=%EXTERNLIBS% -G %GENERATOR% ..
if not x%GENERATOR:Visual Studio=%==x%GENERATOR% (
    echo It contains Visual Studio
    msbuild configureVsCodeSettings.sln
) else (
    echo generator %4
    %GENERATOR%
) 

configureVsCodeSettings.exe %*


