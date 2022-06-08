#OpenCOVER plugins general Browser based on Chromium Embedded Framework
if(NOT WIN32)
  option(BUILD_CEF "build cef for linux64" OFF)
  if(${BUILD_CEF})
    set(CEF_VERSION 98.2.1+g29d6e22+chromium-98.0.4758.109)
    message("building chromium embedded framework version " ${CEF_VERSION})
    message("downloading from https://cef-builds.spotifycdn.com/cef_binary_${CEF_VERSION}_linux64.tar.bz2")
    set(CEF_URL https://cef-builds.spotifycdn.com/cef_binary_${CEF_VERSION}_linux64.tar.bz2)
    set(CEF_CHECKSUM_FILE $ENV{EXTERNLIBS}/cef/hash.sha1)
    file(DOWNLOAD "${CEF_URL}.sha1" ${CEF_CHECKSUM_FILE})
    file(READ "${CEF_CHECKSUM_FILE}" CEF_CHECKSUM)
    include(ExternalProject)
    ExternalProject_Add(cef_build
    PREFIX              $ENV{EXTERNLIBS}/cef
    URL                 ${CEF_URL}
    URL_HASH            SHA1=${CEF_CHECKSUM}
    # DOWNLOAD_NAME       v${VER}.tar.bz2
    # CONFIGURE_COMMAND   CXX=${MPI_CXX_COMPILER} FC=${MPI_Fortran_COMPILER} CC=${MPI_C_COMPILER} ${CMAKE_CURRENT_LIST_DIR}/p4est/configure --prefix=${CMAKE_CURRENT_LIST_DIR} --enable-mpi --enable-mpiio --disable-shared --without-blas
    # BUILD_COMMAND       make
    INSTALL_COMMAND     ${CMAKE_CURRENT_LIST_DIR}/install_cef.sh $ENV{EXTERNLIBS} $ENV{EXTERNLIBS}/cef/src/cef_build
    )
  endif()
endif()

USING(CEF)


SET(HEADERS
  CEF.h
  CEFkeyssymdef.h
)
SET(SOURCES
  CEF.cpp
)
cover_add_plugin(Browser)
set_property(TARGET Browser PROPERTY CXX_STANDARD 14)

if(${BUILD_CEF})
  add_dependencies(Browser cef_build)
endif()

SET(HEADERS
)
SET(SOURCES
  CEFBrowserHelper.cpp
)
if(WIN32)
  SET(SOURCES
    ${SOURCES}
    CEFBrowserHelper.manifest
  )
  ADD_COVISE_EXECUTABLE(CEFBrowserHelper WIN32 ${SOURCES})
else()
  ADD_COVISE_EXECUTABLE(CEFBrowserHelper ${SOURCES})
endif()
COVISE_WNOERROR(CEFBrowserHelper)
set_property(TARGET CEFBrowserHelper PROPERTY CXX_STANDARD 14)
TARGET_LINK_LIBRARIES(CEFBrowserHelper ${EXTRA_LIBS} )

