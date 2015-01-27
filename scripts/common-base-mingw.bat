IF /i "%ARCHSUFFIX%" == "mingwopt" (
  set USE_OPT_LIBS=1
)

if not defined SWIG_HOME  (
   set "SWIG_HOME=%EXTERNLIBS%/swig"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/swig"
)
set "SWIG=%SWIG_HOME%/swig"
set "SWIG_INCLUDES=-I%SWIG_HOME%/Lib -I%SWIG_HOME%/Lib/python -I%SWIG_HOME%/Lib/typemaps"

REM if not defined MPI_HOME  (
  REM set "MPI_HOME=%EXTERNLIBS%/MPICH2/lib"
  REM set "MPI_INCPATH=%EXTERNLIBS%/MPICH2/include"
  REM set "PATHADD=%PATHADD%;%EXTERNLIBS%/MPICH2/bin"
  REM set "MPI_LIBS=-L%EXTERNLIBS%/MPICH2/lib -lcxx -lmpi"
  REM set MPI_DEFINES=HAS_MPI
REM )

if not defined OPENSSL_HOME  (
   set "OPENSSL_HOME=%EXTERNLIBS%/openssl"
   set "OPENSSL_INCPATH=%EXTERNLIBS%/openssl/include"
   set "OPENSSL_DEFINES=HAVE_SSL"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/openssl/bin"
   set "OPENSSL_LIBS=-L%EXTERNLIBS%/openssl/lib -leay32 -lssl32"
)

if not defined GSOAP_HOME (
   set "GSOAP_HOME=%EXTERNLIBS%/gsoap/gsoap"
   set "GSOAP_IMPORTDIR=%EXTERNLIBS%/gsoap/gsoap"
   set "GSOAP_INCPATH=%EXTERNLIBS%/gsoap/gsoap/include"
   set "GSOAP_DEFINES=HAVE_GSOAP"
   set "GSOAP_BINDIR=%EXTERNLIBS%/gsoap/gsoap/bin/win32"
)

if not defined COLLADA_HOME  (
   set "COLLADA_HOME=%EXTERNLIBS%/collada"
   set "COLLADA_INCPATH=%EXTERNLIBS%/collada/include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/collada/lib"
   set "COLLADA_LIBS=-L%EXTERNLIBS%/collada/lib -llibcollada14dom21"
)

if not defined  OPENSCENEGRAPH_HOME (
   set "OPENSCENEGRAPH_HOME=%EXTERNLIBS%/OpenSceneGraph"
   set "OPENSCENEGRAPH_INCPATH=%EXTERNLIBS%/OpenSceneGraph/include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/OpenSceneGraph/bin"
   set "OPENSCENEGRAPH_LIBS=-L%EXTERNLIBS%/OpenSceneGraph/lib -losg -losgDB -losgUtil -losgViewer -losgParticle -losgText -losgSim -losgGA -losgFX -lOpenThreads"
)

set OSGVER=%OPENSCENEGRAPH_HOME%/bin/osgversion
for /f %%v in ('%OSGVER% --version-number') do @set OSG_VER_NUM=%%v
for /f %%v in ('%OSGVER% --so-number') do @set OSG_SO_VER=%%v
for /f %%v in ('%OSGVER% --openthreads-soversion-number') do @set OSG_OT_SO_VER=%%v
if not defined OSG_LIBRARY_PATH (
   set "OSG_LIBRARY_PATH=%OPENSCENEGRAPH_HOME%/bin/osgPlugins-%OSG_VER_NUM%"
)

if not defined DXSDK_HOME (
   IF not defined DXSDK_DIR (
     REM DXSDK_DIR is not set ! Try in EXTERNLIBS?
     set "DXSDK_HOME=%EXTERNLIBS%/dxsdk"
     set "DXSDK_INCPATH=%EXTERNLIBS%/dxsdk/include"
     set "DXSDK_LIBS=-L%EXTERNLIBS%/dxsdk/lib -L%EXTERNLIBS%/dxsdk/lib/x86 -ldxguid -ldxerr9 -ldinput8 -lcomctl32"
   ) ELSE (
     REM DXSDK_DIR is set so use it
     echo DXSDK_DIR is set to "%DXSDK_DIR%"
     set "DXSDK_HOME=%DXSDK_DIR%"
     set "DXSDK_INCPATH=%DXSDK_DIR%include"
     set "DXSDK_LIBS=-L%DXSDK_DIR%lib -L%DXSDK_DIR%lib/x86 -ldxguid -ldxerr9 -ldinput8 -lcomctl32"
   )
)
:DX_LIB_PATH_SET

if not defined ARTOOLKIT_HOME  (
   set HAVE_PTGREY="false"
   set "ARTOOLKIT_HOME=%EXTERNLIBS%/ARToolKit"
   set "ARTOOLKIT_INCPATH=%EXTERNLIBS%/ARToolKit/include %EXTERNLIBS%/videoInput/include"
   set ARTOOLKIT_DEFINES=HAVE_AR
   REM set "PATHADD=%PATHADD%;%EXTERNLIBS%/ARToolKit/lib;%EXTERNLIBS%/videoInput/lib"
   set "ARTOOLKIT_LIBS=-L%EXTERNLIBS%/ARToolKit/lib -L%EXTERNLIBS%/videoInput/lib -lAR -lARMulti -lvideoInput -lstrmbase -lstrmiids -lquartz -lole32 -loleaut32 -luuid"
)

if not defined OSG_HOME (
   set "OSG_HOME=%EXTERNLIBS%/OpenSG"
   set "OSG_INCPATH=%EXTERNLIBS%/OpenSG/include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/OpenSG/lib;%EXTERNLIBS%/tiff/lib"
   set "OSG_LIBS=-L%EXTERNLIBS%/OpenSG/lib OSGBase.lib OSGSystem.lib OSGWindowWIN32.lib"
)


REM if not defined DSIO_HOME (
   REM set "DSIO_HOME=%EXTERNLIBS%/dsio"
   REM set "DSIO_INCPATH=%EXTERNLIBS%/dsio/include"
   REM set "DSIO_DEFINES=HAVE_DSIO"
   REM set "PATHADD=%PATHADD%;%EXTERNLIBS%/dsio/lib"
   REM set "DSIO_LIBS=-L%EXTERNLIBS%/dsio/lib -ldsio20"
REM )


REM if not defined CG_HOME ( 
   REM set "CG_HOME=%EXTERNLIBS%/Cg"
   REM set "CG_INCPATH=%EXTERNLIBS%/Cg/include"
   REM set "PATHADD=%PATHADD%;%EXTERNLIBS%/Cg/bin"
   REM set "CG_LIBS=-L%EXTERNLIBS%/Cg/lib -lcg -lcgGL"
   REM set CG_DEFINES=HAVE_CG
REM )

if not defined PCAN_HOME ( 
   if exist %EXTERNLIBS%/PCAN-Light/nul (
     set "PCAN_HOME=%EXTERNLIBS%/PCAN-Light"
     set "PATHADD=%PATHADD%;%EXTERNLIBS%/PCAN-Light"
     set "PCAN_INCPATH=%EXTERNLIBS%/PCAN-Light/Api"
     set "PCAN_LIBS=-L%EXTERNLIBS%/PCAN-Light/Lib/mingw -lPcan_pci"
     set PCAN_DEFINES=HAVE_PCAN
   )
)

if not defined SIAPP_HOME ( 
   if exist %EXTERNLIBS%/siapp/nul (
     set "SIAPP_HOME=%EXTERNLIBS%/siapp"
     set "PATHADD=%PATHADD%;%EXTERNLIBS%/siapp"
     set "SIAPP_INCPATH=%EXTERNLIBS%/siapp/include"
     set "SIAPP_LIBS=-L%EXTERNLIBS%/siapp/lib -lsiapp"
     set SIAPP_DEFINES=HAVE_SIAPP
   )
)

if not defined FFMPEG_HOME ( 
   if exist %EXTERNLIBS%/ffmpeg/nul (
     set "FFMPEG_HOME=%EXTERNLIBS%/ffmpeg"
     set "PATHADD=%PATHADD%;%EXTERNLIBS%/ffmpeg/bin"
     set "FFMPEG_INCPATH=%EXTERNLIBS%/ffmpeg/include"
     set "FFMPEG_LIBS=-L%EXTERNLIBS%/ffmpeg/lib -lavformat -lavcodec -lswscale -lavutil"
     set FFMPEG_DEFINES=HAVE_FFMPEG_SEPARATE_INCLUDES
   )
)

if not defined QT_HOME ( 
   REM QT_HOME is not set... check QTDIR
   IF not defined QTDIR (
     REM QTDIR is not set ! Try in EXTERNLIBS
     set "QT_HOME=%EXTERNLIBS%/qt"
     set "QT_SHAREDHOME=%EXTERNLIBS%/qt"
     set "QTDIR=%EXTERNLIBS%/qt"
     set "QT_INCPATH=%EXTERNLIBS%/qt/include"
     set "QT_LIBPATH=%EXTERNLIBS%/qt/lib"
     set "PATHADD=%PATHADD%;%EXTERNLIBS%/qt/bin;%EXTERNLIBS%/qt/lib;%EXTERNLIBS%/qwt/lib"
   ) ELSE (
     REM QTDIR is set so try to use it !
     REM Do a simple sanity-check...
     IF NOT EXIST "%QTDIR%/bin/qmake.exe" (
       echo *** WARNING: %QTDIR%/bin/qmake.exe NOT found !
       echo ***          Check QTDIR or simply do NOT set QT_HOME and QTDIR to use the version from EXTERNLIBS!
       pause
     )
     REM Set QT_HOME according to QTDIR. If User ignores any warnings before he will find himself in a world of pain! 
     set "QT_HOME=%QTDIR%"
     set "QT_SHAREDHOME=%QTDIR%"
     set "QT_INCPATH=%QTDIR%/include"
     set "QT_LIBPATH=%QTDIR%/lib"
     set "PATHADD=%PATHADD%;%QTDIR%/bin;%QTDIR%/lib"
   )
)


if not defined ZLIB_HOME  (
   set "ZLIB_HOME=%EXTERNLIBS%/zlib"
   set "ZLIB_INCPATH=%EXTERNLIBS%/zlib/include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/zlib/lib"
   set "ZLIB_LIBS=-L%EXTERNLIBS%/zlib/lib -lzlib"
)

if not defined JPEG_HOME  (
   set "JPEG_HOME=%EXTERNLIBS%/jpeg"
   set "JPEG_INCPATH=%EXTERNLIBS%/jpeg/include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/jpeg/lib"
   set "JPEG_LIBS=-L%EXTERNLIBS%/jpeg/lib -ljpeg"
)

if not defined PNG_HOME (
   set "PNG_HOME=%EXTERNLIBS%/png"
   set "PNG_INCPATH=%EXTERNLIBS%/png/include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/png/lib"
   set "PNG_LIBS=-L%EXTERNLIBS%/png/lib -lpng"
)

if not defined TIFF_HOME (
   set "TIFF_HOME=%EXTERNLIBS%/tiff"
   set "TIFF_INCPATH=%EXTERNLIBS%/tiff/include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/tiff/lib"
   set "TIFF_LIBS=-L%EXTERNLIBS%/tiff/lib -ltiff"
)

if not defined AUDIOFILE_HOME (
   set "AUDIOFILE_HOME=%EXTERNLIBS%/audiofile"
   set "AUDIOFILE_INCPATH=%EXTERNLIBS%/audiofile/include"
   set AUDIOFILE_DEFINES=HAVE_AUDIOFILE
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/audiofile/bin"
   set "AUDIOFILE_LIBS=-L%EXTERNLIBS%/audiofile/lib -laudiofile"
)


if not defined OPENAL_HOME (
   set "OPENAL_HOME=%EXTERNLIBS%/openal"
   set "OPENAL_INCPATH=%EXTERNLIBS%/openal/include"
   set OPENAL_DEFINES=HAVE_OPENAL
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/openal/bin"
   set "OPENAL_LIBS=-L%EXTERNLIBS%/openal/lib -lOpenAL32 -lalut"
)

if not defined COIN3D_HOME  (
   set "COIN3D_HOME=%EXTERNLIBS%/coin3d"
   set "COIN3D_INCPATH=%EXTERNLIBS%/coin3d/include"
   set "COIN3D_DEFINES=COIN_DLL"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/coin3d/bin"
   set "COIN3D_LIBS=-L%EXTERNLIBS%/coin3d/lib -lcoin"
)

if not defined SOQT_HOME  (
   set "SOQT_HOME=%COIN3D_HOME%"
   set "SOQT_INCPATH=%COIN3D_HOME%/include"
   set "PATHADD=%PATHADD%;%COIN3D_HOME%/bin"
   if "%USE_OPT_LIBS%" == "1" (
      set "SOQT_LIBS=-L%COIN3D_HOME%/lib -lsoqt"
   ) else (
      set "SOQT_LIBS=-L%COIN3D_HOME%/lib -lsoqtd"
   )
)


if not defined OIV_HOME (
   set "OIV_HOME=%EXTERNLIBS%/OpenInventor"
   set "OIV_INCPATH=%EXTERNLIBS%/OpenInventor/include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/OpenInventor/lib"
   set "OIV_LIBS=-L%EXTERNLIBS%/OpenInventor/lib -linventor"
)

if not defined GLUT_HOME (
   set "GLUT_HOME=%EXTERNLIBS%/glut"
   set "GLUT_INCPATH=%EXTERNLIBS%/glut/include"
   REM set GLUT_DEFINES=GLUT_NO_LIB_PRAGMA
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/glut/lib"
   set "GLUT_LIBS=-L%EXTERNLIBS%/glut/lib -lglut32"
)

if not defined GLEW_HOME (
   set "GLEW_HOME=%EXTERNLIBS%/glew"
   set "GLEW_INCPATH=%EXTERNLIBS%/glew/include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/glew/bin"
   set "GLEW_LIBS=-L%EXTERNLIBS%/glew/lib -lGLEW32"
)

if not defined VTK_VERSION (
   set "VTK_VERSION=5.4"
)

if not defined VTK_HOME (
   set "VTK_HOME=%EXTERNLIBS%/vtk"
   set "VTK_INCPATH=%EXTERNLIBS%/vtk/include/vtk-%VTK_VERSION%"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/vtk/bin"
   set "VTK_LIBS=-L%EXTERNLIBS%/vtk/lib/vtk-%VTK_VERSION% -lvtkFiltering -lvtkIO -lvtkGraphics -lvtkRendering -lvtkCommon -lvtksys -lgdi32"
)

if not defined ITK_HOME  (
   set "ITK_HOME=%EXTERNLIBS%/itk"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/itk/bin"
   set "ITK_INCPATH=%EXTERNLIBS%/itk/include/InsightToolkit %EXTERNLIBS%/itk/include/InsightToolkit/SpatialObject %EXTERNLIBS%/itk/include/InsightToolkit/Common %EXTERNLIBS%/itk/include/InsightToolkit/Algorithms %EXTERNLIBS%/itk/include/InsightToolkit/Utilities %EXTERNLIBS%/itk/include/InsightToolkit/Numerics %EXTERNLIBS%/itk/include/InsightToolkit/Numerics/Statistics %EXTERNLIBS%/itk/include/InsightToolkit/IO %ITK_INCPATH%;%EXTERNLIBS%/itk/include/InsightToolkit/SpatialObject %EXTERNLIBS%/itk/include/InsightToolkit/BasicFilters %EXTERNLIBS%/itk/include/InsightToolkit/Utilities/vxl/vcl  %EXTERNLIBS%/itk/include/InsightToolkit/Utilities/vxl/core"
   set "ITK_LIBS=-L%EXTERNLIBS%/itk/lib/InsightToolkit -lsnmpapi -lrpcrt4 -lITKEXPAT -litkvcl -litkvnl_algo -litkvnl_inst -lITKIO -lITKDICOMParser -lITKNumerics -lITKAlgorithms -lITKBasicFilters -lITKStatistics -lITKSpatialObject -lITKFEM -litkzlib -litkv3p_netlib -litkvnl -litkgdcm -litkjpeg8 -litkjpeg12 -litkjpeg16 -litkopenjpeg -lITKCommon -litksys -lITKniftiio -lITKznz -lITKNrrdIO -lITKMetaIO -litktiff -litkpng -lgdi32 -lRpcrt4 -lSnmpapi"
)

if not defined TCL_HOME  (
   set "TCL_HOME=%EXTERNLIBS%/TCL"
   set "TCL_INCPATH=%EXTERNLIBS%/TCL/include"
   set "TCL_LIBS=-L%EXTERNLIBS%/TCL/lib -ltcl84"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/TCL/bin"
)

if not defined TK_HOME  ( 
   set "TK_HOME=%EXTERNLIBS%/TCL"
   set "TK_INCPATH=%EXTERNLIBS%/TCL/include"
   set "TK_LIBS=-L%EXTERNLIBS%/TCL/lib -ltk84"
)


if not defined PYTHON_HOME  (
   set "PYTHON_HOME=%EXTERNLIBS%/python"
   set "PYTHON_INCPATH=%EXTERNLIBS%/python/include"
   set "PYTHON_LIBS=-L%EXTERNLIBS%/python/DLLs -lpython26"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/python/DLLs"
   set "PYTHONPATH=%EXTERNLIBS%/python/Lib;%EXTERNLIBS%/python/DLLs"
)

if not defined GDAL_HOME  (
   set "GDAL_HOME=%EXTERNLIBS%/gdal"
   set "GDAL_INCPATH=%EXTERNLIBS%/gdal/include"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/gdal/bin"
   set "GDAL_LIBS=-L%EXTERNLIBS%/gdal/lib -lgdal_i"
)

if not defined XERCESC_HOME  (
   set "XERCESC_HOME=%EXTERNLIBS%/xerces"
   set "XERCESC_INCPATH=%EXTERNLIBS%/xerces/include"
   REM set "XERCESC_DEFINES=XML_LIBRARY"
   set "PATHADD=%PATHADD%;%EXTERNLIBS%/xerces/lib"
   set "XERCESC_LIBS=-L%EXTERNLIBS%/xerces/lib -lxerces-c -lxerces-depdom"
)

REM if not defined WMFSDK_HOME (
   REM set "WMFSDK_HOME=%EXTERNLIBS%/WMFSDK11"
   REM set "WMFSDK_INCPATH=%EXTERNLIBS%/WMFSDK11/Include"
   REM set "WMFSDK_LIBS=-L%EXTERNLIBS%/WMFSDK11/lib -lwmvcore"
   REM set "WMFSDK_DEFINES=HAVE_WMFSDK"
REM )

if not defined CFX5_UNITS_DIR (
   set "CFX5_UNITS_DIR=%COVISEDIR%/icons"
)

set "PTHREAD_LIBS=-lpthread"
set "F77_LIBS=-lgfortran"
set "F90_LIBS=-lgfortran"

rem all path additions have been added to PATHADD
rem as expanding a PATH containing "(x86)" terminates the parentheses opened after if
set "PATH=%PATH%;%QT_HOME%/bin;%PATHADD%"
