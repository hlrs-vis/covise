; files of external dependencies

; note: some external dependencies are not covered here but in the %COVISEDIR%\covise.iss file
; todo: eventually distinguish between the debug and release libs / bins of python to be installed in opt/non-opt COVISE releases
; todo: support more of the optional external dependencies

; files related directly to audiofile (.............license?)
Source: {#EXTERNLIBS}\audiofile\lib\*.dll; DestDir: {#DEXT}\audiofile\lib; Components: runtimes/audiofile
Source: {#EXTERNLIBS}\audiofile\_version.txt; DestDir: {#DEXT}\audiofile; Flags: skipifsourcedoesntexist; Components: runtimes/audiofile

; files related directly to Cg (LGPL-like custom license)
#if (ARCHSUFFIX == "amdwin64") || (ARCHSUFFIX == "amdwin64opt") || (ARCHSUFFIX == "amdwin64opt2008")
Source: {#EXTERNLIBS}\Cg\bin.x64\*.*; DestDir: {#DEXT}\Cg\bin; Components: runtimes/cg
#else
; assume a 32bit Windows
Source: {#EXTERNLIBS}\Cg\bin\*.*; DestDir: {#DEXT}\Cg\bin; Components: runtimes/cg
#endif
Source: {#EXTERNLIBS}\Cg\_version.txt; DestDir: {#DEXT}\Cg; Flags: skipifsourcedoesntexist; Components: runtimes/cg


; files related directly to collada (open XML standard)
Source: {#EXTERNLIBS}\collada\lib\*.dll; DestDir: {#DEXT}\collada\lib; Excludes: *d.dll; Components: runtimes/collada
Source: {#EXTERNLIBS}\collada\_version.txt; DestDir: {#DEXT}\collada; Flags: skipifsourcedoesntexist; Components: runtimes/collada

; files related directly to FreeType (BSD-style FreeType license)
Source: {#EXTERNLIBS}\freetype\lib\*.dll; DestDir: {#DEXT}\freetype\lib; Components: runtimes/freetype
Source: {#EXTERNLIBS}\freetype\_version.txt; DestDir: {#DEXT}\freetype; Flags: skipifsourcedoesntexist; Components: runtimes/freetype

; files related directly to gsoap (LGPL-like Mozilla Public License)
Source: {#GSOAP}\gsoap\bin\win32\*.exe; DestDir: {#DEXT}\gsoap\gsoap\bin\win32; Components: runtimes/gsoap
Source: {#GSOAP}\_version.txt; DestDir: {#DEXT}\gsoap; Flags: skipifsourcedoesntexist; Components: runtimes/gsoap

; files related directly to glew - The OpenGL Extension Wrapper Library (BSD License)
Source: {#EXTERNLIBS}\glew\bin\*.dll; DestDir: {#DEXT}\glew\lib; Components: runtimes/gsoap

; files related directly to JPEG library (LGPL-style license)
Source: {#EXTERNLIBS}\jpeg\lib\*.dll; DestDir: {#DEXT}\jpeg\lib; Components: runtimes/jpeg
Source: {#EXTERNLIBS}\jpeg\_version.txt; DestDir: {#DEXT}\jpeg; Flags: skipifsourcedoesntexist; Components: runtimes/jpeg

; files related directly to libxml2 library (MIT X Consortium license)
Source: {#EXTERNLIBS}\libxml2\lib\*.dll; DestDir: {#DEXT}\libxml2\lib; Components: runtimes/libxml2
Source: {#EXTERNLIBS}\libxml2\_version.txt; DestDir: {#DEXT}\libxml2; Flags: skipifsourcedoesntexist; Components: runtimes/libxml2

; files related directly to OpenAL library (LGPL)
Source: {#EXTERNLIBS}\OpenAL\lib\*.dll; DestDir: {#DEXT}\OpenAL\lib; Excludes: *d.dll; Components: runtimes/openal
Source: {#EXTERNLIBS}\OpenAL\_version.txt; DestDir: {#DEXT}\OpenAL; Flags: skipifsourcedoesntexist; Components: runtimes/openal

; files related directly to OpenCV library (BSD License)
Source: {#EXTERNLIBS}\OpenCV\lib\*.dll; DestDir: {#DEXT}\OpenCV\lib; Excludes: *d.dll; Components: runtimes/opencv
Source: {#EXTERNLIBS}\OpenCV\_version.txt; DestDir: {#DEXT}\OpenCV; Flags: skipifsourcedoesntexist; Components: runtimes/opencv

; files related directly to OpenInventor library (BSD License)
Source: {#EXTERNLIBS}\OpenInventor\lib\*.dll; DestDir: {#DEXT}\OpenInventor\lib; Excludes: *d.dll; Components: runtimes/openinventor
Source: {#EXTERNLIBS}\OpenInventor\_version.txt; DestDir: {#DEXT}\OpenInventor; Flags: skipifsourcedoesntexist; Components: runtimes/openinventor

; files related directly to OpenSceneGraph library (OpenSceneGraph license = modified LGPL)
Source: {#OPENSCENEGRAPH}\bin\*.dll; DestDir: {#DEXT}\OpenSceneGraph-{#OSG_VER_NUM}\bin; Excludes: *d.dll; Components: runtimes/openscenegraph
Source: {#OPENSCENEGRAPH}\bin\*.exe; DestDir: {#DEXT}\OpenSceneGraph-{#OSG_VER_NUM}\bin; Excludes: *d.exe; Components: runtimes/openscenegraph;
Source: {#OPENSCENEGRAPH}\bin\osgplugins-{#OSG_VER_NUM}\*.dll; DestDir: {#DEXT}\OpenSceneGraph-{#OSG_VER_NUM}\bin\osgplugins-{#OSG_VER_NUM}; Components: runtimes/openscenegraph; Flags: skipifsourcedoesntexist
Source: {#OPENSCENEGRAPH}\_version.txt; DestDir: {#DEXT}\OpenSceneGraph; Flags: skipifsourcedoesntexist; Components: runtimes/openscenegraph

; files related directly to OpenSSL library (similar to Apache license)
Source: {#OPENSSL}\bin\*.dll; DestDir: {#DEXT}\OpenSSL\bin; Components: runtimes/openssl
Source: {#OPENSSL}\bin\*.exe; DestDir: {#DEXT}\OpenSSL\bin; Components: runtimes/openssl
Source: {#OPENSSL}\_version.txt; DestDir: {#DEXT}\OpenSSL; Flags: skipifsourcedoesntexist; Components: runtimes/openssl

; files related directly to PNG library (zlib license)
Source: {#PNG}\lib\*.dll; DestDir: {#DEXT}\png\lib; Components: runtimes/png
Source: {#PNG}\_version.txt; DestDir: {#DEXT}\png; Flags: skipifsourcedoesntexist; Components: runtimes/png

; files related directly to pthread library (LGPL)
Source: {#PTHREAD}\lib\*.dll; DestDir: {#DEXT}\pthreads\lib; Components: runtimes/pthreads
Source: {#PTHREAD}\_version.txt; DestDir: {#DEXT}\pthreads; Flags: skipifsourcedoesntexist; Components: runtimes/pthreads

; files related directly to Python (Python license http://www.python.org/psf/license/)
Source: {#PYTHON}\DLLs\*.dll; DestDir: {#DEXT}\Python\DLLs; Components: runtimes/python
Source: {#PYTHON}\DLLs\*.exe; DestDir: {#DEXT}\Python\DLLs; Components: runtimes/python
;Source: {#PYTHON}\DLLs\*.ico; DestDir: {#DEXT}\Python\DLLs; Components: runtimes/python
Source: {#PYTHON}\DLLs\*.pyd; DestDir: {#DEXT}\Python\DLLs; Components: runtimes/python
; todo: check, if all libs in {#Python}\Lib\site-packages\ can be shipped in that form
Source: {#PYTHON}\Lib\*.*; DestDir: {#DEXT}\Python\Lib; Flags: recursesubdirs; Components: runtimes/python
; if the following files do not exist, they should exist in {#Python}\DLLs
Source: {#PYTHON}\python*.exe; DestDir: {#DEXT}\Python; Flags: skipifsourcedoesntexist; Components: runtimes/python
Source: {#PYTHON}\w9xpopen*.exe; DestDir: {#DEXT}\Python; Flags: skipifsourcedoesntexist; Components: runtimes/python
Source: {#PYTHON}\_version.txt; DestDir: {#DEXT}\Python; Flags: skipifsourcedoesntexist; Components: runtimes/python

; files related directly to Qt (LGPL or commercial)
Source: {#QT}\bin\*.dll; DestDir: {#DEXT}\qt\bin; Components: runtimes/qt
Source: {#QT}\bin\*.exe; DestDir: {#DEXT}\qt\bin; Excludes: *d4.dll;  Components: runtimes/qt
;Source: {#QT}\bin\qt.conf; DestDir: {#DEXT}\qt\bin; Components: runtimes/qt
Source: {#QT}\plugins\accessible\*.dll; DestDir: {#DEXT}\qt\plugins\accessible; Excludes: *d4.dll, *d.dll; Flags: recursesubdirs skipifsourcedoesntexist; Components: runtimes/qt
Source: {#QT}\plugins\codecs\*.dll; DestDir: {#DEXT}\qt\plugins\codecs; Excludes: *d4.dll, *d.dll; Flags: recursesubdirs skipifsourcedoesntexist; Components: runtimes/qt
Source: {#QT}\plugins\designer\*.dll; DestDir: {#DEXT}\qt\plugins\designer; Excludes: *d4.dll, *d.dll; Flags: recursesubdirs skipifsourcedoesntexist; Components: runtimes/qt
Source: {#QT}\plugins\graphicssystems\*.dll; DestDir: {#DEXT}\qt\plugins\graphicssystems; Excludes: *d4.dll, *d.dll; Flags: recursesubdirs skipifsourcedoesntexist; Components: runtimes/qt
Source: {#QT}\plugins\iconengines\*.dll; DestDir: {#DEXT}\qt\plugins\iconengines; Excludes: *d4.dll, *d.dll; Flags: recursesubdirs skipifsourcedoesntexist; Components: runtimes/qt
Source: {#QT}\plugins\imageformats\*.dll; DestDir: {#DEXT}\qt\plugins\imageformats; Excludes: *d4.dll, *d.dll; Flags: recursesubdirs skipifsourcedoesntexist; Components: runtimes/qt
Source: {#QT}\plugins\script\*.dll; DestDir: {#DEXT}\qt\plugins\script; Excludes: *d4.dll, *d.dll; Flags: recursesubdirs skipifsourcedoesntexist; Components: runtimes/qt
Source: {#QT}\plugins\sqldrivers\*.dll; DestDir: {#DEXT}\qt\plugins\sqldrivers; Excludes: *d4.dll, *d.dll; Flags: recursesubdirs skipifsourcedoesntexist; Components: runtimes/qt
Source: {#QT}\phrasebooks\*.qph; DestDir: {#DEXT}\qt\phrasebooks; Flags: skipifsourcedoesntexist; Components: runtimes/qt
Source: {#QT}\translations\*.qm; DestDir: {#DEXT}\qt\translations; Components: runtimes/qt; Flags: skipifsourcedoesntexist
Source: {#QT}\translations\*.ts; DestDir: {#DEXT}\qt\translations; Components: runtimes/qt; Flags: skipifsourcedoesntexist
Source: {#QT}\_version.txt; DestDir: {#DEXT}\qt; Flags: skipifsourcedoesntexist; Components: runtimes/qt

; files related directly to SWIG (SWIG license http://www.swig.org/copyright.html)
Source: {#EXTERNLIBS}\swig\*.*; DestDir: {#DEXT}\swig; Flags: recursesubdirs; Components: runtimes/swig
Source: {#EXTERNLIBS}\swig\_version.txt; DestDir: {#DEXT}\swig; Flags: skipifsourcedoesntexist; Components: runtimes/swig

; files related directly to Tcl/Tk library (LGPL-like http://www.tcl.tk/software/tcltk/license_terms.txt)
Source: {#EXTERNLIBS}\Tcl\bin\*.*; DestDir: {#DEXT}\Tcl\bin; Components: runtimes/tcl
Source: {#EXTERNLIBS}\Tcl\glut\lib\glut.dll; DestDir: {#DEXT}\Tcl\glut\lib; Flags: skipifsourcedoesntexist; Components: runtimes/tcl
Source: {#EXTERNLIBS}\Tcl\_version.txt; DestDir: {#DEXT}\Tcl; Flags: skipifsourcedoesntexist; Components: runtimes/tcl

; files related directly to tiff library (LGPL-like?)
Source: {#TIFF}\lib\tiff.dll; DestDir: {#DEXT}\tiff\lib; Components: runtimes/tiff
Source: {#TIFF}\_version.txt; DestDir: {#DEXT}\tiff; Flags: skipifsourcedoesntexist; Components: runtimes/tiff

; files related directly to UnixUtils tools (GPL, which is ok, since scripts using UnixUtils are 
; supplied as source)
Source: {#EXTERNLIBS}\UnixUtils\*.exe; DestDir: {#DEXT}\UnixUtils; Components: runtimes/unixutils
Source: {#EXTERNLIBS}\UnixUtils\_version.txt; DestDir: {#DEXT}\UnixUtils; Flags: skipifsourcedoesntexist; Components: runtimes/unixutils
Source: {#EXTERNLIBS}\sed\*.exe; DestDir: {#DEXT}\sed; Components: runtimes/sed
Source: {#EXTERNLIBS}\sed\_version.txt; DestDir: {#DEXT}\sed; Flags: skipifsourcedoesntexist; Components: runtimes/sed

; files related directly to xerces library (Apache Software License 2.0 http://www.apache.org/licenses/LICENSE-2.0.html)
Source: {#XERCES}\lib\*.dll; DestDir: {#DEXT}\xerces\lib; Excludes: *d.dll; Components: runtimes/xerces
Source: {#XERCES}\_version.txt; DestDir: {#DEXT}\xerces; Flags: skipifsourcedoesntexist; Components: runtimes/xerces

; files related directly to zlib library (zlib)
Source: {#ZLIB}\lib\*.dll; DestDir: {#DEXT}\zlib\lib; Excludes: *d.dll; Components: runtimes/zlib
Source: {#ZLIB}\_version.txt; DestDir: {#DEXT}\zlib; Flags: skipifsourcedoesntexist; Components: runtimes/zlib

; files related directly to ffmpeg
Source: {#EXTERNLIBS}\ffmpeg\bin\*.dll; DestDir: {#DEXT}\ffmpeg\bin; Components: runtimes/ffmpeg

; files related directly to abaqus
#if (DISTRO_TYPE == "PLAINVANILLA")
Source: {#EXTERNLIBS}\abaqus\lib\*.dll; DestDir: {#DEXT}\abaqus\lib; Components: runtimes/abaqus
Source: {#EXTERNLIBS}\abaqus\_version.txt; DestDir: {#DEXT}\abaqus; Flags: skipifsourcedoesntexist; Components: runtimes/abaqus
#endif

;#if (DISTRO_TYPE != "RTT") && (DISTRO_TYPE != "KLSM") 
; files related directly to jt (DAIMLER ONLY)
;Source: {#EXTERNLIBS}\jt\lib\*.dll; DestDir: {#DEXT}\jt\lib; Components: runtimes/jt
;#endif

; files related directly to ZIP library 
; CC
; Source: {#EXTERNLIBS}\7zip\*.*; DestDir: {#DEXT}\7zip; Components: runtimes/zlib
; Source: {#EXTERNLIBS}\7zip\_version.txt; DestDir: {#DEXT}\7zip; Flags: skipifsourcedoesntexist; Components: runtimes/zlib

