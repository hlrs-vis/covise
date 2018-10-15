here I try to write down how to compile all necessary dependencies for windows

everything is done from a normal covise zebu shell if not otherwise mentioned
Visual Studio 2015, Cmake and Tortoise Git are installed

#ZLIB
Downlload zlib 1.2.8
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/zlib

#JPEG
Download jpeg-9
devenv jpeg.sln
swtich to x64; compile Debug and Release
Manually copy jpegD.lib and jpeg.lib to c:/src/externlibs/zebu/jpeg/lib
copy include file to jpeg/include

#ffmpeg
get builds from https://ffmpeg.zeranoe.com/builds/
copy shared libs and dev files to c:/src/externlibs/zebu/ffmpeg

#freetype
get freetype 2.5.3
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/freetype

#inventor or coin, see below
get SGI Inventor
change include directories and lib directories to externlibs/freetype and zlib
replace abs with fabs in source code
remove #ifndef Win32 from image.h
compile debug and release and manually copy to externlibs/zebu/inventor

#Coin 3.1.3
set COINDIR=c:\src\externlibs\zebu\coin3d
compile all four versions. It gets installed in COINDIR

#OpenEXR
git clone https://github.com/openexr/openexr.git
add this to CMakeList.txt
SET(CMAKE_DEBUG_POSTFIX "d")
in both ilmbase and openexr/openexr
copy dlls to all after building and installing ilmbase
dlls are in lib, not in bin as they should be
and exclude examples and tests
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenEXR -DZLIB_ROOT=c:/src/externlibs/zebu/zlib -DILMBASE_PACKAGE_PREFIX=c:/src/externlibs/zebu/OpenEXR
#SDL
get SDL 2.0.5 from https://www.libsdl.org/download-2.0.php
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/SDL -DCMAKE_DEBUG_POSTFIX=d
#CURL
get latest CURL; go to winbuild
nmake /F Makefile.vc VC=14 WITH_DEVEL=c:\src\externlibs\zebu\zlib WITH_ZLIB=dll MACHINE=x64 MODE=dll
nmake /F Makefile.vc VC=14 WITH_DEVEL=c:\src\externlibs\zebu\zlib WITH_ZLIB=dll MACHINE=x64 MODE=dll DEBUG=yes
rename debug build of curl.exe to curl_debug.exe ; manually copy builds to externlibs:
copy lib/libcurl_debug to libcurld.lib
builds\libcurl-vc14-x64-release-dll-zlib-dll-ipv6-sspi-winssl
#giflib
get giflib windows version from http://blog.issamsoft.com/?q=en/node/82
fix x64 options: add d to General/Target Name
and _MS_VISUAL_STUDIO to C++/Preprocessor/defines
compile and manually copy to externlibs
#tiff
got tiff-3.7.3
converted old project file compiled and copied libs from objs directory to externlibs
#png
use provided solution
convert to new vs format
adjust zlib source dir in zlib.prop
copy zconf.h from build2015 directory to zlib dir so that it is found by libpng
add x64 platform
add D suffix to debug Target Name
only compile libpng
copy to externlibs
#glut copied binaries from tamarau
#nvtt
cloned from here: https://github.com/castano/nvidia-texture-tools.git
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/nvtt -DCMAKE_DEBUG_POSTFIX=d

#xerces-c
download from apache site and build from provided solutions
manually copy from Build directory to externlibs

#ICU
download icu
go to source/allinone
build solution and copy to externlibs

#Perl
http://www.activestate.com/activeperl/

#Python
got python from python.org
got externals with pcbuild/get_externals.bat
compiled pcbuild.sln Debug and Release
amd64/*.lib to libs amd64/*.dll *.exe *.pdb to bin
Lib to Lib and include to include
copy PC/pyconfig.h to include
uncomment the following in pyconfig.h
//#				pragma comment(lib,"python35_d.lib")
#			elif defined(Py_LIMITED_API)
#				pragma comment(lib,"python3.lib")
#			else
//#				pragma comment(lib,"python35.lib")
#OpenSSL
use openssl from python externals
PATH=D:\src\gitbase\Python-3.5.2\externals\nasm-2.11.06;%PATH%
perl Configure VC-WIN64A
ms\do_win64a

perl util\mk1mf.pl debug dll VC-WIN64A >ms\ntdebugdll.mak
edit ntdebugdll.mak and add D to dll names 
nmake -f ms\ntdll.mak

#qt
set PATH=c:\src\externlibs\zebu\Python2\bin;%PATH%
set PYTHONHOME=c:\src\externlibs\zebu\..\shared\Python2;c:\src\externlibs\zebu\Python2
configure -prefix c:/src/externlibs/zebu/qt5 -opensource -debug-and-release -nomake tests -make libs -make tools -nomake examples -nomake tests -confirm-license -openssl -I c:/src/externlibs/zebu/OpenSSL/include  -icu -I c:/src/externlibs/zebu/icu/include -L c:/src/externlibs/zebu/icu/lib -openssl-linked  -L C:/src/externlibs/zebu/OpenSSL/lib -openssl -openssl-linked OPENSSL_LIBS="-lUser32 -lAdvapi32 -lGdi32" OPENSSL_LIBS_DEBUG="-lssleay32D -llibeay32D" OPENSSL_LIBS_RELEASE="-lssleay32 -llibeay32" -platform win32-msvc2015 -mp -opengl dynamic -angle
nmake
nmake install

#SoQT
get old soqt Version 1.4.1
set COIN3D_HOME=c:\src\externlibs\zebu\coin3d
set COINDIR=c:\src\externlibs\zebu\coin3d
added QT_NO_OPENGL_ES_2;QT_NO_OPENGL_ES defines to all configurations
added #undef QT_OPENGL_ES_2 to qopengl.h
remove spwinput_x11.c from project

#new soqt:
hg clone
hg clone https://bitbucket.org/Coin3D/simage
hg clone https://bitbucket.org/Coin3D/coin
hg clone https://bitbucket.org/Coin3D/soqt
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/simage -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal
add zlib.lib to link line
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/Coin3D -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/simage;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal
disable BUILD_DOCUMENTATION and TESTS
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/SoQt -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/simage;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal


#proj.4
git clone https://github.com/OSGeo/proj.4.git
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/proj4 -DCMAKE_DEBUG_POSTFIX=d
#gdal
edit nmake.opt adjust Python path

PROJ_FLAGS = -DPROJ_STATIC
PROJ_INCLUDE = -ID:\src\externlibs\zebu\proj4\local\include
PROJ_LIBRARY = D:\src\externlibs\zebu\proj4\local\lib

!IFDEF DEBUG
GDALLIB	=    $(GDAL_ROOT)/gdal_i_d.lib
!ELSE
!IFDEF DLLBUILD
GDALLIB	=    $(GDAL_ROOT)/gdal_i.lib
!ELSE
GDALLIB	=    $(GDAL_ROOT)/gdal.lib
!ENDIF
!ENDIF

!IFDEF DEBUG
GDAL_DLL =	gdal$(VERSION)D.dll
!ELSE
GDAL_DLL =	gdal$(VERSION).dll
!ENDIF

CURL

GDAL_HOME = "C:\src\externlibs\zebu\gdal"

nmake /f makefile.vc MSVC_VER=1900 WIN64=YES
nmake /f makefile.vc MSVC_VER=1900 WIN64=YES install
manually copy .lib files to externlibs/gdal/lib
nmake /f makefile.vc clean
nmake /f makefile.vc MSVC_VER=1900 WIN64=YES DEBUG=1 WITH_PDB=1
nmake /f makefile.vc MSVC_VER=1900 WIN64=YES DEBUG=1 WITH_PDB=1 install
rename lib file to gdalD.lib and gdalD_i.lib and copy to externlibs/gdal/lib
#PThreads
download pthreads4w

comment out this in _ptw32.h
#  define int64_t _int64
#  define uint64_t unsigned _int64

#OpenSceneGraph
Clone https://github.com/openscenegraph/OpenSceneGraph.git to OpenSceneGraph
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenSceneGraph -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal
adjust project file:
for tiff: add c:/src/externlibs/zebu/jpeg/lib/jpeg[d].lib
gdal and ogr: use gdalD_i.lib for the debug version
nvtt Debug: (Release without d )
C:\src\externlibs\zebu\nvtt\lib\static\nvttd.lib
C:\src\externlibs\zebu\nvtt\lib\static\nvcored.lib
C:\src\externlibs\zebu\nvtt\lib\static\nvimaged.lib
C:\src\externlibs\zebu\nvtt\lib\static\nvmathd.lib
C:\src\externlibs\zebu\nvtt\lib\static\squishd.lib
C:\src\externlibs\zebu\nvtt\lib\static\nvthreadd.lib
C:\src\externlibs\zebu\nvtt\lib\static\bc7d.lib
C:\src\externlibs\zebu\nvtt\lib\static\bc6hd.lib

#boost
bootstrap.bat

the following goes to user-config.jam, than add --user-config=user-config.jam or add it to project-config.jam
# Configure specific Python version.

using python : 3.5 : /src/externlibs/zebu/python : /src/externlibs/zebu/python/include : /src/externlibs/zebu/python/libs : <python-debugging>on : _d ;
#using python : 3.5 : /src/externlibs/zebu/python : /src/externlibs/zebu/python/include : /src/externlibs/zebu/python/libs : <define>BOOST_DEBUG_PYTHON=1 : _d ;
# first make with <define> then with <python-debugging>on , or try to do <python-debugging>on <define>BOOST_DEBUG_PYTHON=1

#using zlib : 1.2.8  : <include>c:/src/externlibs/zebu/zlib/include <search>c:/src/externlibs/zebu/zlib/lib ;

#using mpi ;

####b2 address-model=64 --build-type=complete --prefix=c:\src\externlibs\zebu\boost --build-dir=build  variant=debug,release link=static,shared threading=multi runtime-link=shared
#specifying zlib binaries does not work, specifying a source dir does:
set ZLIB_SOURCE=d:\src\gitbase\zlib-1.2.8
b2 address-model=64 architecture=x86 --prefix=c:\src\externlibs\zebu\boost --build-dir=build  variant=debug,release link=static,shared threading=multi runtime-link=shared --without-mpi -j8  --debug-configuration -d+2 --user-config=user-config.jam


#OpenCV3
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenCV3
set contrib/modules directory in cmake-gui
disable performance tests and normal tests, build (be very patient) and install

#vtk
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/VTK  -DCMAKE_DEBUG_POSTFIX=d

#hdf5
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/hdf5

#alvar
alvar needs openCV 2.4 (currently 2412)
thus first compile OpenCV2.4 as follows:
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenCV2
then
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/alvar -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/alvar -DALVAR_NOGLUT=true -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

#osgcal
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/osgcal -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/cal3d
in core_osgCal Debug change cal3d.lib to cal3d_d.lib
in applications add C:/src/externlibs/zebu/osgCal/lib to library directories 
and cal3d_d.lib to input

#assimp
https://github.com/assimp/assimp.git
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/assimp -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

#citygml
https://github.com/jklimke/libcitygml.git
cmake .. -G "Visual Studio 14 2015 Win64" -DXERCESC_STATIC=false -DLIBCITYGML_STATIC_CRT=false -DLIBCITYGML_OSGPLUGIN=true -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/citygml -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/OpenSceneGraph

#osgEphemeris
https://github.com/hlrs-vis/osgephemeris.git
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/osgEphemeris -DCMAKE_DEBUG_POSTFIX=d
-DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/OpenSceneGraph

#OpenVR
https://github.com/ValveSoftware/openvr.git
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenVR -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv
change the cmake files to SHARED instead of STATIC
add WIN64; to the defines in the project file
copy the header files from the source dir to externlibs

#v8
set PYTHONHOME=
set DEPOT_TOOLS_WIN_TOOLCHAIN=0
mkdir depot_tools
cd depot_tools
curl -O https://storage.googleapis.com/chrome-infra/depot_tools.zip
cmake -E tar xf "depot_tools.zip" --format=zip
SET PATH=%CD%;%CD%\python276_bin;%PATH%
cd ..
gclient config https://chromium.googlesource.com/v8/v8
set GYP_MSVS_VERSION=2015
gclient sync
cd v8
python tools/dev/v8gen.py x64.release
ninja -C out.gn/x64.release

#fftw3 
Download source from http://www.fftw.org/download.html and extract.
Additional we need from http://www.fftw.org/install/windows.html Project files to compile FFTW 3.3 with Visual Studio 2010 under Building FFTW 3.x under Visual C++/Intel compilers (scroll down). Extract this archive into the above folder SDKROOT\fftw-3.3.4.
Open project file SDKROOT\fftw-3.3.4\fftw-3.3-libs\fftw-3.3-libs.sln.
Select Release and Win32 or x64.
Open context menu on libfftw-3.3, select Properties. Under Configuration Properties > General check Platform toolset. Change from Windows 7.1SDK to Visual Studio 2013 (v120). Go to Configuration Properties > C/C++ > Code Generation > Run time library and change to Multi threaded DLL (/MD). Close dialog with Ok.
Open context menu on libfftw-3.3, select Add > Existing item, select file SDKROOT\fftw-3.3.4\api\mkprinter-str.c
Now build project libfftw-3.3

#geos
# do not get tar geos from https://trac.osgeo.org/geos/ but use git otherwise cmake will fail
git clone https://git.osgeo.org/gogs/geos/geos.git

cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/geos -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv
#osgEarth
fix toplevel cmakeLists.txt 
FIND_PACKAGE(ZLIB)
set(ZLIB_LIBRARY ${ZLIB_LIBRARY_RELEASE})

cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/osgEarth -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv


#SISL
cmake .. -G "Visual Studio 14 2015 Win64" -Dsisl_INSTALL_PREFIX=c:/src/externlibs/zebu/sisl -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv


#e57
change the following in CmakeLists.txt:


# Override flags to enable prepare for linking to static runtime
#set(CMAKE_USER_MAKE_RULES_OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/c_flag_overrides.cmake)
#set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cxx_flag_overrides.cmake)

#set(Boost_USE_STATIC_LIBS   ON)
#set(Boost_USE_STATIC_RUNTIME ON)

#set(Xerces_USE_STATIC_LIBS On)

#add_definitions(-DBOOST_ALL_NO_LIB )#-DXERCES_STATIC_LIBRARY)
  add_definitions("-DBOOST_ALL_NO_LIB")
  add_definitions("-DBOOST_ALL_DYN_LINK")
  
add_library( E57RefImpl STATIC
    src/refimpl/E57Foundation.cpp
    src/refimpl/E57FoundationImpl.cpp
    src/refimpl/E57FoundationImpl.h
    src/refimpl/E57Simple.cpp
    src/refimpl/E57SimpleImpl.cpp
    src/refimpl/E57SimpleImpl.h
    src/time_conversion/time_conversion.c
    include/E57Foundation.h
)



download E57SimpleImpl source and add E57Simple.* to E57RefImpl

cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/e57 -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

 


###########
###########
#Lamure

cmake .. -G "Visual Studio 14 2015 Win64" -DBOOST_ROOT=c:/src/externlibs/zebu/boost -DGLOBAL_EXT_DIR=c:/src/externlibs/zebu -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/lamure -DCMAKE_DEBUG_POSTFIX=d -DSCHISM_INCLUDE_SEARCH_DIR=D:/src/gitbase/schism -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

add -DSCM_STATIC_BUILD

add to lamure_rendering and _app

C:\src\externlibs\zebu\schism\lib\scm_core-gdd.lib
C:\src\externlibs\zebu\schism\lib\scm_gl_core-gdd.lib
C:\src\externlibs\zebu\schism\lib\scm_gl_util-gdd.lib
C:\src\externlibs\zebu\freeglut\lib\x64\freeglut.lib
C:\src\externlibs\zebu\schism\lib\scm_input-gdd.lib
opengl32.lib

#cgal
https://github.com/CGAL/cgal.git
cmake .. -G "Visual Studio 14 2015 Win64" -DBOOST_ROOT=c:/src/externlibs/zebu/boost -DGLOBAL_EXT_DIR=c:/src/externlibs/zebu -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/cgal -DCMAKE_DEBUG_POSTFIX=d -DSCHISM_INCLUDE_SEARCH_DIR=D:/src/gitbase/schism -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv


#libqtviewer
qmake -t vclib libQGLViewer-2.7.0.pro -spec win32-msvc2015
devenv libQGLViewer-2.vcxproj

#schism
got to build/cmake/build
add compiler to schism_compiler.cmake
add boost search path in schism_boost.cmake
SET(SCM_BOOST_INCLUDE_SEARCH_DIRS
    /src/externlibs/zebu/boost/include/boost-1_62

SET(SCM_BOOST_LIBRARY_SEARCH_DIRS
    /src/externlibs/zebu/boost/lib
cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/schism -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/lamure -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv
# freeimage
download freeimage from webpage or sourcecforge
replace libwebp source by current git repository
git clone https://github.com/webmproject/libwebp.git
on windows open the visual studio project and add enc/backward_references_cost_enc.cal3d and dsp/ssim*.c
Build debug and release


########
#gmsh
for linking OCC set CASROOT
































#########################################
#########################################
### UWP
#########################################
#########################################

cmake -G "Visual Studio 15 2017"  -DCMAKE_SYSTEM_NAME:STRING="WindowsStore" -DCMAKE_SYSTEM_VERSION:STRING="10.0" -DOSG_BUILD_PLATFORM_UWP:BOOL=ON -DOPENGL_PROFILE:STRING=GLES2 -DOSG_WINDOWING_SYSTEM:STRING=NONE -DOSG_USE_UTF8_FILENAME:BOOL=ON -DDYNAMIC_OPENSCENEGRAPH:BOOL=OFF -DDYNAMIC_OPENTHREADS:BOOL=OFF -DBUILD_OSG_APPLICATIONS:BOOL=OFF -DBUILD_OSG_EXAMPLES:BOOL=OFF -DOPENGL_INCLUDE_DIR:PATH="c:/src/externlibs/uwp/angle/include" -DOPENGL_HEADER1:STRING="#include <GLES2/gl2.h>" -DOPENGL_gl_LIBRARY:STRING="c:/src/externlibs/uwp/angle/libGLESv2.lib" -DEGL_INCLUDE_DIR:PATH="c:/src/externlibs/uwp/angle/include" -DEGL_LIBRARY:STRING="c:/src/externlibs/uwp/angle/lib/libEGL.lib" ..
