REM @echo off

call vcpkg-settings.bat

set "vc=%VCPKG_ROOT%\vcpkg"



REM choco -y install cmake --installargs 'ADD_CMAKE_TO_PATH=""System""'
REM choco -y install git swig winflexbison

"%vc%" install assimp curl[ssl] glew giflib libpng tiff xerces-c zlib libjpeg-turbo
"%vc%" install vtk gdcm hdf5[cpp]
"%vc%" install netcdf-cxx4 itk
"%vc%" install pthreads tbb libmicrohttpd python3
"%vc%" install osg osgearth
"%vc%" install ffmpeg opencv
"%vc%" install proj4 gdal libgeotiff
"%vc%" install boost-asio boost-bimap boost-chrono boost-date-time boost-mpl boost-program-options boost-serialization boost-signals2 boost-smart-ptr boost-uuid boost-variant boost-interprocess boost-iostreams
"%vc%" install qt5-tools qt5-base qt5-svg
"%vc%" install openvr
"%vc%" install openexr
"%vc%" install pcl
"%vc%" install libarchive libzip snappy
"%vc%" install embree3
"%vc%" install eigen3
"%vc%" install msmpi
"%vc%" install fftw3 sdl2 freeimage
"%vc%" install libusb
"%vc%" install cfitsio
REM for visionaray
"%vc%" install freeglut
REM "%vc%" install openexr