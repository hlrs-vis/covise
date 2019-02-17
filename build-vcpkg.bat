REM @echo off

if "%VCPKG_ROOT%" EQU "" (
    set VCPKG_ROOT=C:\vcpkg
)

if NOT EXIST "%VCPKG_ROOT%\.vcpkg-root" (
    VCPKG_ROOT has to be set to the root directory of your vcpkg installation
    goto END
)

set "vcdir=%VCPKG_ROOT%"
set VCPKG_DEFAULT_TRIPLET=x64-windows
set "generator=Visual Studio 15 2017 Win64"

set EXTERNLIBS=
set ARCHSUFFIX=vcpkgopt
set "vc=%vcdir%\vcpkg"
set "COVISEDIR=%CD%"
set "COVISEDESTDIR=%CD%"

set cfg=RelWithDebInfo
if "%ARCHSUFFIX%" == "vcpkg" (
    set cfg=Debug
)
set ARCHSUFFIX=vcpkg

set "verb=/consoleloggerparameters:Summary;Verbosity=minimal;ForceNoAlign;ShowTimestamp"
set "par=/m"

REM choco -y install cmake --installargs 'ADD_CMAKE_TO_PATH=""System""'
REM choco -y install git swig winflexbison

"%vc%" install assimp curl freeglut glew giflib libpng tiff xerces-c zlib libjpeg-turbo
"%vc%" install vtk gdcm2 hdf5[cpp]
"%vc%" install pthreads tbb libmicrohttpd python3
"%vc%" install osg
"%vc%" install ffmpeg opencv
"%vc%" install proj4 gdal libgeotiff
"%vc%" install boost-asio boost-bimap boost-chrono boost-date-time boost-mpl boost-program-options boost-serialization boost-signals2 boost-smart-ptr boost-uuid boost-variant boost-interprocess
"%vc%" install qt5-tools qt5-base qt5-svg
"%vc%" install openvr
"%vc%" install openexr
"%vc%" install pcl
"%vc%" install libarchive libzip snappy
"%vc%" install embree3

%vc% list
REM %vc% integrate project

mkdir vcpkg
cd vcpkg

:COVISE
mkdir build.covise
cd build.covise
REM cmake -G "%generator%" "-DCMAKE_TOOLCHAIN_FILE=%vcdir%\scripts\buildsystems\vcpkg.cmake" -DCOVISE_BUILD_RENDERER:BOOL=OFF -DCOVISE_USE_VISIONARAY:BOOL=OFF ../..
cmake -G "%generator%" "-DCMAKE_TOOLCHAIN_FILE=%vcdir%\scripts\buildsystems\vcpkg.cmake" -DCOVISE_BUILD_RENDERER:BOOL=OFF ../..
if errorlevel 1 goto UPDIR
msbuild /m covise.sln /p:Configuration=%cfg% %verb% %par%
if errorlevel 1 goto UPDIR

:UPDIR
cd ..
cd ..

:END
