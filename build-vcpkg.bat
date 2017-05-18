REM @echo off

if "%VCPKG_ROOT%" EQU "" (
    VCPKG_ROOT has to be set
    goto END
)

set "vcdir=%VCPKG_ROOT%"

set EXTERNLIBS=
set ARCHSUFFIX=vcpkg
set "vc=%vcdir%\vcpkg"
set "COVISEDIR=%CD%"
set "COVISEDESTDIR=%CD%"

REM choco -y install cmake --installargs 'ADD_CMAKE_TO_PATH=""System""'
REM choco -y install git swig winflexbison

%vc% install assimp:x64-windows boost:x64-windows curl:x64-windows freeglut:x64-windows glew:x64-windows giflib:x64-windows libpng:x64-windows qt5:x64-windows tiff:x64-windows xerces-c:x64-windows zlib:x64-windows libjpeg-turbo:x64-windows vtk:x64-windows  
%vc% install pthreads:x64-windows tbb:x64-windows

%vc% list
REM %vc% integrate project

mkdir %ARCHSUFFIX%
cd %ARCHSUFFIX%
mkdir build.covise
cd build.covise
cmake -G "Visual Studio 15 2017 Win64" "-DCMAKE_TOOLCHAIN_FILE=%vcdir%\scripts\buildsystems\vcpkg.cmake" -DCOVISE_BUILD_RENDERER:BOOL=OFF ../..
msbuild /m covise.sln

"%vc%" install osg:x64-windows

cd ..
mkdir build.cover
cd build.cover
cmake -G "Visual Studio 15 2017 Win64" "-DCMAKE_TOOLCHAIN_FILE=%vcdir%\scripts\buildsystems\vcpkg.cmake" ../../src/OpenCOVER
msbuild /m OpenCOVER.sln

:END
