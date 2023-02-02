REM @args: covise dir, archsuffix, dependencyPath, Qt version, generator, overwrite
set COVISEDIR=%1
set ARCHSUFFIX=%2
set EXTERNLIBS=%3
set QT_VERSION=%4
set GENERATOR=%5
call %COVISEDIR%/winenv.bat %ARCHSUFFIX% %QT_VERSION%
Rem write down covise environment to reuse in vs code debugger settings and CMake wrapper
set > %COVISEDIR%/.vscode/configureCovise/covise.env
cd %COVISEDIR%/.vscode/configureCovise

set "compiled="
if exist %~dp0/build/configureVsCodeSettings.exe set compiled=1
if exist %~dp0/build/Debug\configureVsCodeSettings.exe set compiled=1
if not defined compiled (
    rmdir /s /q build
    mkdir build
    cd build
    echo generator %generator%
    cmake -DCMAKE_PREFIX_PATH=%EXTERNLIBS% -G %GENERATOR% ..
    cmake --build .
    if exist configureVsCodeSettings.exe (
        configureVsCodeSettings.exe %*
    ) else if exist Debug\ (
        Debug\configureVsCodeSettings.exe %*
    )
)

if exist %EXTERNLIBS%\stow.bat if NOT exist %EXTERNLIBS%\all call %EXTERNLIBS%\stow.bat
