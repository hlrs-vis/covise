# @file
# 
# CMakeLists.txt for renderer QtRenderer
#
# @author Blasius Czink

PROJECT(WebGL CXX)

USING(PNG)
USING(FREETYPE)
USING(OPENTHREADS)
USING(QT)
USING(MICROHTTPD)

# look for png12/png.h, has to be in PNG_INCLUDE_DIR, otherwise USING(PNG) won't work correctly
find_path(PNG12_INCLUDE_DIR "libpng12/png.h"
  NO_DEFAULT_PATH
  PATHS
  ${PNG_PNG_INCLUDE_DIR}
  DOC "libpng 1.2 headers"
)

if (NOT PNG12_INCLUDE_DIR)
   message("WebGL: libpng is not version 1.2")
   return()
endif()

QT_ADD_RESOURCES(WSI_RESOURCE_FILES "../../../share/covise/web/WebGL.qrc" OPTIONS "-no-compress")
SET(HEADERS
  debug.h  ft.h  md5.h  WebGLRenderer.h
)

SET(SOURCES
  WebGLRenderer.cpp debug.cpp ft.cpp md5.cpp
)

ADD_COVISE_MODULE(Renderer WebGL  ${WSI_RESOURCE_FILES})
# setup what we need from Qt
qt_use_modules(WebGL Widgets)


TARGET_LINK_LIBRARIES(WebGL
  coAppl coCore coUtil
  ${EXTRA_LIBS}
)

COVISE_INSTALL_TARGET(WebGL)
