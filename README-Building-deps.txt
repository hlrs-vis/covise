here I try to write down how to compile all necessary dependencies for windows

everything is done from a normal covise zebu shell if not otherwise mentioned
Visual Studio 2015, Cmake and Tortoise Git are installed

#COVISE
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/netcdf

#nlohmann/json
git clone https://github.com/nlohmann/json.git
cd json
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/nlohmann_json 


#Botan
create C:\src\externlibs\zebu\botan
git clone https://github.com/randombit/botan
cd botan 
python configure.py --cc=msvc --os=windows --debug-mode --msvc-runtime=MDd --library-suffix=d --prefix=C:\src\externlibs\zebu\botan
nmake
nmake check
nmake install
python configure.py --cc=msvc --os=windows --msvc-runtime=MD --prefix=C:\src\externlibs\zebu\botan
nmake
nmake check
nmake install


#ZLIB
Downlload zlib 1.2.8
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/zlib

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
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/freetype

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
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenEXR -DZLIB_ROOT=c:/src/externlibs/zebu/zlib -DILMBASE_PACKAGE_PREFIX=c:/src/externlibs/zebu/OpenEXR
#SDL
get SDL 2.0.5 from https://www.libsdl.org/download-2.0.php
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/SDL -DCMAKE_DEBUG_POSTFIX=d
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
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/nvtt -DCMAKE_DEBUG_POSTFIX=d

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

#OpenSSL 1.1.1
Download from https://www.openssl.org/source/
install strawberry perl and NASM via installer and make shure they are in PATH
From admin vs development prompt 
perl Configure VC-WIN64A --prefix=c:/src/externlibs/zebu/OpenSSL --openssldir=c:/src/externlibs/zebu/OpenSSL
nmake
nmake test
nmake install

#qt5
set PATH=c:\src\externlibs\zebu\Python2\bin;%PATH%
set PYTHONHOME=c:\src\externlibs\zebu\..\shared\Python2;c:\src\externlibs\zebu\Python2
configure -prefix c:/src/externlibs/zebu/qt5 -opensource -debug-and-release -make libs -make tools -nomake examples -nomake tests -confirm-license -openssl -I c:/src/externlibs/zebu/OpenSSL/include  -icu -I c:/src/externlibs/zebu/icu/include -L c:/src/externlibs/zebu/icu/lib -openssl-linked  -L C:/src/externlibs/zebu/OpenSSL/lib -openssl -openssl-linked OPENSSL_LIBS="-lUser32 -lAdvapi32 -lGdi32" OPENSSL_LIBS_DEBUG="-lssleay32D -llibeay32D" OPENSSL_LIBS_RELEASE="-lssleay32 -llibeay32" -platform win32-msvc2015 -mp -opengl dynamic -angle
nmake
nmake install

#qt6 following https://wiki.qt.io/Building_Qt_6_from_Git
fix compress_files.js C:\src\gitbase\qt6\qtwebengine\src\3rdparty\chromium\third_party\devtools-frontend\src\scripts\build\compress_files.js
serialize file processing to fix compression
  for(i=0;i<files.length;i++)
  {
	  await compressFile(files[i]);
  }
  
use admin developer cmd
cd build
set PATH=c:\src\externlibs\zebu\Python2\bin;%PATH%
##..\configure.bat -prefix c:\src\externlibs\zebu\qt6 -skip qtspeech -debug-and-release -qt-zlib -openssl-linked -- -D OPENSSL_ROOT_DIR=C:/src/externlibs/zebu/OpenSSL
..\configure -prefix c:/src/externlibs/zebu/qt6 -opensource -debug-and-release -make tools -nomake examples -nomake tests -confirm-license -openssl  -icu -openssl-linked -opengl dynamic -- -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/md4c;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/flex;c:/src/externlibs/zebu/bison;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal
cmake --build . --parallel
ninja qtdeclarative
ninja
ninja install

#md4c
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/md4c -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/simage;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal

#3mx
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/3mx -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/simage;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal


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
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/simage -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal
add zlib.lib to link line
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/Coin3D -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/simage;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal
disable BUILD_DOCUMENTATION and TESTS
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/SoQt -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/simage;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal

#jsbsim
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/jsbsim -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/simage;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal
#sqlite3
cl sqlite3.c -link -dll -out:sqlite3.dll
cl sqlite3.c -link -dll -debug  -out:sqlite3d.dll
dumpbin /exports sqlite3.dll > exports.txt
echo LIBRARY SQLITE3 > sqlite3.def
echo EXPORTS >> sqlite3.def
for /f "skip=19 tokens=4" %A in (exports.txt) do echo %A >> sqlite3.def
lib /def:sqlite3.def /out:sqlite3.lib
dumpbin /exports sqlite3d.dll > exportsd.txt
echo LIBRARY SQLITE3D > sqlite3d.def
echo EXPORTS >> sqlite3d.def
for /f "skip=19 tokens=4" %A in (exportsd.txt) do echo %A >> sqlite3d.def
lib /def:sqlite3d.def /out:sqlite3d.lib
mkdir c:\src\externlibs\zebu\sqlite3
mkdir c:\src\externlibs\zebu\sqlite3\bin
mkdir c:\src\externlibs\zebu\sqlite3\lib
mkdir c:\src\externlibs\zebu\sqlite3\include
copy *.h  c:\src\externlibs\zebu\sqlite3\include
copy *.lib  c:\src\externlibs\zebu\sqlite3\lib
copy *.dll  c:\src\externlibs\zebu\sqlite3\bin
copy *.pdb  c:\src\externlibs\zebu\sqlite3\bin
cl shell.c sqlite3.c -Fesqlite3.exe
copy sqlite3.exe c:\src\externlibs\zebu\sqlite3\bin

static libs:

cl /c /EHsc /DEBUG sqlite3.c
rename sqlite3.obj sqlite3d.obj
lib sqlite3d.obj
cl /c /EHsc sqlite3.c
lib sqlite3.obj
copy *.lib c:\src\externlibs\zebu\sqlite3\lib

#proj.4
git clone https://github.com/OSGeo/proj.4.git
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/proj4 -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/sqlite3;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/curl -DCMAKE_DEBUG_POSTFIX=d
cmake-gui
disable testing and tiff

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
nmake /f makefile.vc MSVC_VER=1900 WIN64=YES devinstall
# (done by devinstallmanually copy .lib files to externlibs/gdal/lib
nmake /f makefile.vc clean
change PROJ_LIBRARY to debug in nmake.opt
PROJ_LIBRARY = C:/src/externlibs/zebu/proj4/lib/\proj_d.lib shell32.lib ole32.lib
nmake /f makefile.vc MSVC_VER=1900 WIN64=YES DEBUG=1 WITH_PDB=1
nmake /f makefile.vc MSVC_VER=1900 WIN64=YES DEBUG=1 WITH_PDB=1 install
nmake /f makefile.vc MSVC_VER=1900 WIN64=YES DEBUG=1 WITH_PDB=1 devinstall
rename lib file to gdalD.lib and gdalD_i.lib and copy to externlibs/gdal/lib
#PThreads
download pthreads4w

comment out this in _ptw32.h
#  define int64_t _int64
#  define uint64_t unsigned _int64

#OpenSceneGraph
Clone https://github.com/openscenegraph/OpenSceneGraph.git to OpenSceneGraph
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenSceneGraph -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal
cmake ..  -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenSceneGraph -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal


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
//set ZLIB_SOURCE=d:\src\gitbase\zlib-1.2.8
//b2 install address-model=64 architecture=x86 --prefix=c:\src\externlibs\zebu\boost --build-dir=build  variant=debug,release link=static,shared threading=multi runtime-link=shared --without-python --without-mpi -j8  --debug-configuration -d+2
b2 install address-model=64 architecture=x86 --prefix=c:\src\externlibs\zebu\boost --build-dir=build  variant=debug,release link=static,shared threading=multi runtime-link=shared --with-zlib --without-python --without-mpi -j8  --debug-configuration -d+2 -sZLIB_LIBRARY="c:\src\externlibs\zebu\zlib\lib" -sZLIB_INCLUDE="c:\src\externlibs\zebu\zlib\include"

## boost.Python and Boost.mpi not needed anymore, thus don't do this:

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
b2 address-model=64 architecture=x86 --prefix=c:\src\externlibs\zebu\boost --build-dir=build  variant=debug,release link=static,shared threading=multi runtime-link=shared --without-mpi -j8  --debug-configuration python-debugging=on -d+2

#pybind11:
Then edit \path\to\python\include\pybind11\detail\common.h. Remove these blocks:

About line 106

#  if defined(_DEBUG)
#    define PYBIND11_DEBUG_MARKER
#    undef _DEBUG
#  endif
About line 131

#  if defined(PYBIND11_DEBUG_MARKER)
#    define _DEBUG
#    undef PYBIND11_DEBUG_MARKER
#  endif
Now then, Microsoft has safe APIs for strdup and sscanf. If you rename these in pybind11 code, the crashes on quit disappear.

Edit pybind11.h and replace strdup with _strdup. Edit detail\common.h and replace sscanf with sscanf_s.


#OpenCV3
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenCV3
set contrib/modules directory in cmake-gui
disable performance tests and normal tests, build (be very patient) and install

#OpenCV4
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenCV4  -DCMAKE_DEBUG_POSTFIX=d -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules
disable performance tests and normal tests, build (be very patient) and install

#vtk
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/VTK  -DCMAKE_DEBUG_POSTFIX=d

#hdf5
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/hdf5

#alvar
alvar needs openCV 2.4 (currently 2412)
thus first compile OpenCV2.4 as follows:
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenCV2
then
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/alvar -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/alvar -DALVAR_NOGLUT=true -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

#osgcal
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/osgcal -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/cal3d
in core_osgCal Debug change cal3d.lib to cal3d_d.lib
in applications add C:/src/externlibs/zebu/osgCal/lib to library directories 
and cal3d_d.lib to input

#Bullet
cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=false -DUSE_MSVC_RUNTIME_LIBRARY_DLL=true -DUSE_MSVC_AVX=true -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/bullet -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/cal3d

#osgWorks
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/osgWorks -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Bullet;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/cal3d

#osgBullet
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/osgbullet -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/osgWorks/lib;c:/src/externlibs/zebu/osgWorks;c:/src/externlibs/zebu/Bullet;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/cal3d  -DBulletInstallType="Alternate Install Location" -DBulletInstallLocation=c:/src/externlibs/zebu/bullet

#GDCM
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_DEBUG_POSTFIX=d  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/gdcm -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/cal3d


#assimp
https://github.com/assimp/assimp.git
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/assimp -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

#citygml
https://github.com/jklimke/libcitygml.git
cmake .. -G "Visual Studio 17 2022" -A x64  -DXERCESC_STATIC=false -DLIBCITYGML_STATIC_CRT=false -DLIBCITYGML_OSGPLUGIN=true -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/citygml -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/OpenSceneGraph

#osgEphemeris
https://github.com/hlrs-vis/osgephemeris.git
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/osgEphemeris -DCMAKE_DEBUG_POSTFIX=d
-DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/OpenSceneGraph

#OpenVR
https://github.com/ValveSoftware/openvr.git
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenVR -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv
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

cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/geos -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv
#osgEarth
fix toplevel cmakeLists.txt 
FIND_PACKAGE(ZLIB)
set(ZLIB_LIBRARY ${ZLIB_LIBRARY_RELEASE})

cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/osgEarth -DPROTOBUF_USE_DLLS=true -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/glew;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/tinyxml2;c:/src/externlibs/zebu/sqlite3


#SISL
cmake .. -G "Visual Studio 17 2022" -A x64  -Dsisl_INSTALL_PREFIX=c:/src/externlibs/zebu/sisl -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

#libarchive
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/libarchive -DCMAKE_DEBUG_POSTFIX=d -DENABLE_WERROR=OFF  -DENABLE_TEST=OFF -DENABLE_COVERAGE=OFF -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

#Protobuf https://github.com/protocolbuffers/protobuf.git
git submodule update --init --recursive
cd cmake
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/protobuf -Dprotobuf_MSVC_STATIC_RUNTIME=TRUE -Dprotobuf_BUILD_SHARED_LIBS=TRUE -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

#OSI https://github.com/OpenSimulationInterface/open-simulation-interface.git
#git submodule update --init --recursive
#cd cmake
add    option cc_enable_arenas = true;        to all .proto files
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OSI -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/protobuf -DProtobuf_INCLUDE_DIR=c:/src/externlibs/zebu/protobuf/include  -DProtobuf_PROTOC_EXECUTABLE=c:/src/externlibs/zebu/protobuf/bin/protoc.exe -DProtobuf_LIBRARIES=c:/src/externlibs/zebu/protobuf/lib
add PROTOBUF_USE_DLLS
ADD_DEFINITIONS("-DPROTOBUF_USE_DLLS")

#hdf5
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/hdf5 -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv
cmake-gui .. disable building tests
ensure that zlib is enabled
#NETCDF
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/netcdf -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5
cmake-gui .. because the above fails but creates a valid CMakeCache
I manually needed to set CMAKE_DEBUG_POTFIX to d
#cfitsio
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/cfitsio -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5

#cgns
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/cgns -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/netcdf

#cudpp
"c:\Program Files\CMake\bin"\cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/cudpp -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/netcdf

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

cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/e57 -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

 


###########
###########
#Lamure

cmake .. -G "Visual Studio 17 2022" -A x64  -DBOOST_ROOT=c:/src/externlibs/zebu/boost -DGLOBAL_EXT_DIR=c:/src/externlibs/zebu -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/lamure -DCMAKE_DEBUG_POSTFIX=d -DSCHISM_INCLUDE_SEARCH_DIR=D:/src/gitbase/schism -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv

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
cmake .. -G "Visual Studio 17 2022" -A x64  -DBOOST_ROOT=c:/src/externlibs/zebu/boost -DGLOBAL_EXT_DIR=c:/src/externlibs/zebu -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/cgal -DCMAKE_DEBUG_POSTFIX=d -DSCHISM_INCLUDE_SEARCH_DIR=D:/src/gitbase/schism -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv


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
	
cd build\cmake\build
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/schism -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/freeimage

set BOOST_ROOT=c:/src/externlibs/zebu/boost
cd D:\src\gitbase\lamure\build\build
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/lamure -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/freeglut;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/glfw;c:/src/externlibs/zebu/freeimage;c:/src/externlibs/zebu/cgal;c:/src/externlibs/zebu/gmp;c:/src/externlibs/zebu/mpfr;c:/src/externlibs/zebu/glew;c:/src/externlibs/zebu/glm
manualy changed glut to freeglut in cmake-gui

https://github.com/glfw/glfw
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/glfw -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism

glm:
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/glm -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism

# freeimage
download freeimage from webpage or sourcecforge
replace libwebp source by current git repository
git clone https://github.com/webmproject/libwebp.git
on windows open the visual studio project and add enc/backward_references_cost_enc.cal3d and dsp/ssim*.c
Build debug and release

##############
# pyqt5 build sip first


############
# sip
set PYTHONHOME=c:\src\externlibs\zebu\Python
python configure.py
nmake
nmake install


########
#gmsh
for linking OCC set CASROOT


pcl:
cmake .. -G "Visual Studio 17 2022" -A x64 -DBOOST_ROOT=c:/src/externlibs/zebu/boost -DGLOBAL_EXT_DIR=c:/src/externlibs/zebu -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/pcl -DCMAKE_DEBUG_POSTFIX=d -DSCHISM_INCLUDE_SEARCH_DIR=D:/src/gitbase/schism -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/eigen;c:/src/externlibs/zebu/flann;c:/src/externlibs/zebu/qhull;c:/src/externlibs/zebu/png;c:/src/externlibs/zebu/vtk
flann
cmake .. -G "Visual Studio 17 2022" -A x64 -DBOOST_ROOT=c:/src/externlibs/zebu/boost -DGLOBAL_EXT_DIR=c:/src/externlibs/zebu -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/flann -DCMAKE_DEBUG_POSTFIX=d -DSCHISM_INCLUDE_SEARCH_DIR=D:/src/gitbase/schism -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/eigen
qhull
cmake .. -G "Visual Studio 17 2022" -A x64 -DBOOST_ROOT=c:/src/externlibs/zebu/boost -DGLOBAL_EXT_DIR=c:/src/externlibs/zebu -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/qhull -DCMAKE_DEBUG_POSTFIX=d -DSCHISM_INCLUDE_SEARCH_DIR=D:/src/gitbase/schism -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/eigen


############################
# speex
https://www.speex.org/downloads/
download speex and speexdsp
go to win32 dir, convert solutions and add d suffix to the debug project
create x64 config
compile Release_SSD2
change runtime to MD and MDd
manually copy to lib and include



#xdmf
git clone https://gitlab.kitware.com/xdmf/xdmf.git
#don't: copy libxml2static.lib to libxml2.lib so that the standard find finds the lib (should be already done in the latest externlibs
#add  LIBXML_STATIC=TRUE to DEFINES for all projects
#add C:\src\externlibs\zebu\libxml2\lib\libxml2staticD.lib to additionalLibraries
cmake .. -G "Visual Studio 17 2022" -A x64 -DBOOST_ROOT=c:/src/externlibs/zebu/boost -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/xdmf -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/libxml2;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/iconv



#SUMO
set SWIG_DIR=c:\externlibs\zebu\swig
cmake ..  -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/sumo -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/proj4;c:/src/externlibs/zebu/gprc;c:/src/externlibs/zebu/fox


##########################
#### libWebrtc
###cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/libwebrtc  -DCMAKE_DEBUG_POSTFIX=d

#google webrtc
#checkout source
set PATH=c:\src\gitbase\depot_tools;%PATH%
set DEPOT_TOOLS_WIN_TOOLCHAIN=0
set GYP_MSVS_VERSION=2019

mkdir webrtc-checkout
cd webrtc-checkout
gclient
fetch --nohooks webrtc
gclient sync
cd src
gn gen --ide=vs2019 out\x64 --filters=//:webrtc "--args=is_debug=false use_lld=false is_clang=false rtc_include_tests=false rtc_build_tools=true rtc_win_video_capture_winrt=true rtc_build_examples=false rtc_win_use_mf_h264=true enable_libaom=false rtc_enable_protobuf=false"
cd out\x64
all.sln

#freeopcua
cmake ..  -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/freeopcua -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/xml2;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/proj4;c:/src/externlibs/zebu/gprc;c:/src/externlibs/zebu/fox
#open62541
cmake ..  -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/open62541 -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/xml2;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/sphinx;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/proj4;c:/src/externlibs/zebu/gprc;c:/src/externlibs/zebu/fox


###################################
#####VSG
###############################

#glslang glslang that comes with vulkan sdk does not provide a debug version
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/glslang -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/osi;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/gtest


#VulkanSceneGraph VSG
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/VulkanSceneGraph -DVSG_MAX_DEVICES=4 -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/glslang;C:\VulkanSDK\1.2.170.0;c:/src/externlibs/zebu/glslang;c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/osi;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/gtest

#vsgGIS
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/vsgGIS  -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/VulkanSceneGraph;c:/src/externlibs/zebu/assimp;c:/src/externlibs/zebu/glslang;C:\VulkanSDK\1.2.170.0;c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/osi;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/gtest

#vsgXchange
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/vsgXchange  -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/vsgGIS;c:/src/externlibs/zebu/VulkanSceneGraph;c:/src/externlibs/zebu/assimp;c:/src/externlibs/zebu/glslang;C:\VulkanSDK\1.2.170.0;c:/src/externlibs/zebu/glslang;c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/osi;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/gtest

#vsgImGui
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/vsgImGui  -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/vsgGIS;c:/src/externlibs/zebu/vsgXchange;c:/src/externlibs/zebu/VulkanSceneGraph;c:/src/externlibs/zebu/assimp;c:/src/externlibs/zebu/glslang;C:\VulkanSDK\1.2.170.0;c:/src/externlibs/zebu/glslang;c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/osi;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/gtest

#vsgExamples
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/vsgExamples  -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/vsgImGui;c:/src/externlibs/zebu/vsgGIS;c:/src/externlibs/zebu/vsgXchange;c:/src/externlibs/zebu/VulkanSceneGraph;c:/src/externlibs/zebu/assimp;c:/src/externlibs/zebu/glslang;C:\VulkanSDK\1.2.170.0;c:/src/externlibs/zebu/glslang;c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/osi;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/gtest


#########################################
#########################################
### UWP
#########################################
#########################################

cmake .. -G "Visual Studio 17 2022" -A x64   -DCMAKE_SYSTEM_NAME:STRING="WindowsStore" -DCMAKE_SYSTEM_VERSION:STRING="10.0" -DOSG_BUILD_PLATFORM_UWP:BOOL=ON -DOPENGL_PROFILE:STRING=GLES2 -DOSG_WINDOWING_SYSTEM:STRING=NONE -DOSG_USE_UTF8_FILENAME:BOOL=ON -DDYNAMIC_OPENSCENEGRAPH:BOOL=OFF -DDYNAMIC_OPENTHREADS:BOOL=OFF -DBUILD_OSG_APPLICATIONS:BOOL=OFF -DBUILD_OSG_EXAMPLES:BOOL=OFF -DOPENGL_INCLUDE_DIR:PATH="c:/src/externlibs/uwp/angle/include" -DOPENGL_HEADER1:STRING="#include <GLES2/gl2.h>" -DOPENGL_gl_LIBRARY:STRING="c:/src/externlibs/uwp/angle/libGLESv2.lib" -DEGL_INCLUDE_DIR:PATH="c:/src/externlibs/uwp/angle/include" -DEGL_LIBRARY:STRING="c:/src/externlibs/uwp/angle/lib/libEGL.lib" ..

#########################################
Google Test
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/gtest -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/osi;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/boost
#########################################

OpenPass
#cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/openpass -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/osi;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/gtest;c:/src/externlibs/zebu/OSI;c:/src/externlibs/zebu/fmi
#Linux: . ~/bin/gcc9; cmake .. -DWITH_GUI=false -DWITH_TESTS=false -DCMAKE_INSTALL_PREFIX=/data/extern_libs/rhel79/OpenPASS07 CMAKE_PREFIX_PATH=/data/extern_libs/rhel79/protobuf:/data/extern_libs/rhel79/fmi/:/data/extern_libs/rhel79/boost1_74:/data/extern_libs/rhel79/OpenSimulationInterface
cmake .. -G "Visual Studio 17 2022" -A x64 -DBoost_USE_STATIC_LIBS=OFF -DBUILD_SHARED_LIBS=ON -DWITH_GUI=true -DWITH_TESTS=false -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/openpass -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/osi;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/gtest;c:/src/externlibs/zebu/OSI;c:/src/externlibs/zebu/fmi


FMI Library  https://github.com/modelon-community/fmi-library
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/fmi -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/GEOS;c:/src/externlibs/zebu/V8;c:/src/externlibs/zebu/osi;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/hdf5;c:/src/externlibs/zebu/protobuf;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/gtest




#######
DART
#########

eigen3 // disable tests after cmake
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/eigen -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/OpenSceneGraph
libccd
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/ccd -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/OpenSceneGraph
fcl Flexible collision library // disable tests after cmake
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/fcl -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/ccd
octomap
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/octomap -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/ccd

dart
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/dart -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/eigen3;c:/src/externlibs/zebu/ccd;c:/src/externlibs/zebu/fcl;c:/src/externlibs/zebu/assimp;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/octomap;c:/src/externlibs/zebu/tinyxml2
manually add C:\src\externlibs\zebu\glut\include (too lazy to fix the cmake build)


###############
rbdl
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/rbdl -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/eigen3;c:/src/externlibs/zebu/ccd;c:/src/externlibs/zebu/fcl;c:/src/externlibs/zebu/assimp;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/octomap;c:/src/externlibs/zebu/tinyxml2


###############
https://github.com/TheComet/ik
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/cometIK -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/eigen3;c:/src/externlibs/zebu/ccd;c:/src/externlibs/zebu/fcl;c:/src/externlibs/zebu/assimp;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/octomap;c:/src/externlibs/zebu/tinyxml2


##########
IFC++
https://github.com/ifcquery/ifcplusplus.git
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/ifcpp -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/eigen3;c:/src/externlibs/zebu/ccd;c:/src/externlibs/zebu/fcl;c:/src/externlibs/zebu/assimp;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/octomap;c:/src/externlibs/zebu/tinyxml2

##########
openNURBS
https://github.com/mcneel/opennurbs
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/openNURBS -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Coin3D;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/eigen3;c:/src/externlibs/zebu/ccd;c:/src/externlibs/zebu/fcl;c:/src/externlibs/zebu/assimp;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/octomap;c:/src/externlibs/zebu/tinyxml2

##########
osgPhysX
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/osgPhysX -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/PhysX;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/eigen3;c:/src/externlibs/zebu/ccd;c:/src/externlibs/zebu/fcl;c:/src/externlibs/zebu/assimp;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/octomap;c:/src/externlibs/zebu/tinyxml2

##########
cef
download from
https://cef-builds.spotifycdn.com/index.html#windows64
cd c:/src/externlibs/zebu/cef/build
cmake .. -G "Visual Studio 17 2022" -A x64  -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/cef -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/PhysX;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/ffmpeg;c:/src/externlibs/zebu/freetype;c:/src/externlibs/zebu/giflib;c:/src/externlibs/zebu/glut;c:/src/externlibs/zebu/icu;c:/src/externlibs/zebu/jpeg;c:/src/externlibs/zebu/libpng;c:/src/externlibs/zebu/nvtt;c:/src/externlibs/zebu/OpenEXR;c:/src/externlibs/zebu/OpenSSL;c:/src/externlibs/zebu/Python;c:/src/externlibs/zebu/qt5;c:/src/externlibs/zebu/SDL;c:/src/externlibs/zebu/tiff;c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/opencv;c:/src/externlibs/zebu/schism;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/eigen3;c:/src/externlibs/zebu/ccd;c:/src/externlibs/zebu/fcl;c:/src/externlibs/zebu/assimp;c:/src/externlibs/zebu/boost;c:/src/externlibs/zebu/octomap;c:/src/externlibs/zebu/tinyxml2
change runtime library to Debug DLL or DLL -->in visual studio right click libcef_dll_wrapper->properties->c/c++->Code Generation->Runtime Library to Md/MDd
remove _HAS_ITERATOR_DEBUGGING=0 from preprocessor definitions in the debug config ->properties->c/c++ -> Preprocessor -> Preprocessor Definitions -> remove _HAS_ITERATOR_DEBUGGING=0
Build 
copy Release/libcef.ddl or Deubu/libcef.dll and contets of Resources to externlibs/zebu/all

########
vrmlexp
edit covise/src/cmake/Find3DSMAX.cmake add 20xx version
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/Cal3d;c:/src/externlibs/zebu/curl;c:/src/externlibs/zebu/xerces

#####
u3d https://github.com/ningfei/u3d
cmake .. -G "Visual Studio 17 2022" -A x64  -DU3D_SHARED:BOOL=ON -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/u3d -DCMAKE_DEBUG_POSTFIX=d -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/xerces;c:/src/externlibs/zebu/gdal;c:/src/externlibs/zebu/OpenSceneGraph;c:/src/externlibs/zebu/zlib;c:/src/externlibs/zebu/png;c:/src/externlibs/zebu/jpeg


###
vistle
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/botan;c:/src/externlibs/zebu/proj4;c:/src/externlibs/zebu/zsd



###
fltk
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/fltk -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/botan;c:/src/externlibs/zebu/proj4;c:/src/externlibs/zebu/zsd

###
freealut
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/alut -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/botan;c:/src/externlibs/zebu/proj4;"c:/Progra~2/OpenAL 1.1 SDK"



###
Open Audio Server
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=c:/src/externlibs/zebu/OpenAS -DCMAKE_PREFIX_PATH=c:/src/externlibs/zebu/fltk;c:/src/externlibs/zebu/mxml;"c:/Progra~2/OpenAL 1.1 SDK";c:/src/externlibs/zebu/alut

