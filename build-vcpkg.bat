REM @echo off

if "%VCPKG_ROOT%" EQU "" (
    VCPKG_ROOT has to be set
    goto END
)

set "vcdir=%VCPKG_ROOT%"
set VCPKG_DEFAULT_TRIPLET=x64-windows
set "generator=Visual Studio 15 2017 Win64"

set EXTERNLIBS=
set ARCHSUFFIX=vcpkg
set "vc=%vcdir%\vcpkg"
set "COVISEDIR=%CD%"
set "COVISEDESTDIR=%CD%"

REM choco -y install cmake --installargs 'ADD_CMAKE_TO_PATH=""System""'
REM choco -y install git swig winflexbison

%vc% install assimp boost curl freeglut glew giflib libpng qt5 tiff xerces-c zlib libjpeg-turbo vtk  
%vc% install pthreads tbb libmicrohttpd python3

%vc% list
REM %vc% integrate project

mkdir %ARCHSUFFIX%
cd %ARCHSUFFIX%
mkdir build.covise
cd build.covise
cmake -G "%generator%" "-DCMAKE_TOOLCHAIN_FILE=%vcdir%\scripts\buildsystems\vcpkg.cmake" -DCOVISE_BUILD_RENDERER:BOOL=OFF ../..
msbuild /m covise.sln

"%vc%" install osg:%vct%

cd ..
mkdir build.cover
cd build.cover
cmake -G "%generator%" "-DCMAKE_TOOLCHAIN_FILE=%vcdir%\scripts\buildsystems\vcpkg.cmake" ../../src/OpenCOVER
msbuild /m OpenCOVER.sln

:END
