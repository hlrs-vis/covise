REM @args: covise dir, archsuffix, dependencyPath, Qt version, generator, overwrite
set EXTERNLIBS=%3
set VCPKG_ROOT=%3

set COVISEDIR=%1
call %1/winenv.bat %2 %4
Rem write down covise environment to reuse in vs code debugger settings and CMake wrapper
set > %1/.vscode/configureCovise/covise.env
cd %1/.vscode/configureCovise
rmdir /s /q build
mkdir build
cd build
set GENERATOR=%5

cmake -DCMAKE_PREFIX_PATH=%EXTERNLIBS% -G %GENERATOR% ..
cmake --build .

if exist configureVsCodeSettings.exe (
    configureVsCodeSettings.exe %*
) else if exist Debug\ (
    Debug\configureVsCodeSettings.exe %*
)

if exist %EXTERNLIBS%\stow.bat if NOT exist %EXTERNLIBS%\all call %EXTERNLIBS%\stow.bat

