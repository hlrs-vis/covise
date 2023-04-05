# @file
#
# CMakeLists.txt for covise renderer OpenCOVER
#
# @author Blasius Czink

#COVISE_UNFINISHED()

PROJECT(OpenCOVER C CXX)
# this is already set in the top most CMakeLists.txt set (CMAKE_CXX_STANDARD 11)
if(NOT COVISE_OPENCOVER_INTERNAL_PROJECT)

    set(COVISE_EXPORTS_FILE "cover-exports.cmake")
    set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
endif()

set(COVISE_BUILDING_COVER TRUE)

if(NOT COVISE_OPENCOVER_INTERNAL_PROJECT)
    option(BUILD_SHARED_LIBS "Build shared libraries" ON)
    option(COVISE_USE_CUDA "Use CUDA" ON)
    option(COVISE_USE_MPI "Use MPI" ON)
    option(COVISE_USE_VIRVO "Use Virvo volume rendering library" ON)
    option(COVISE_WARNING_IS_ERROR "Treat warnings as errors" ON)
    option(COVISE_USE_FOLDERS "Enable solution folders in Visual Studio. Disable for Express versions." ON)
    option(COVISE_EXPORT_TO_INSTALL "Place COVISE exported targets cmake-file in <install directory>/<archsuffix>/lib (default). Otherwise in build directory." ON)
    option(COVISE_USE_OSMESA "Use OSMesa OpenGL library for CPU based off-screen rendering" OFF)
endif()

option(COVER_ENABLE_VOIP "Enable VoIP linphone client plugin" OFF)
mark_as_advanced(COVER_ENABLE_VOIP)

macro (GETENV_PATH var name)
   set(${var} $ENV{${name}})
   if (WIN32)
      string(REGEX REPLACE "\\\\" "/" ${var} "${${var}}")
   endif(WIN32)
endmacro (GETENV_PATH var name)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(PLUGIN_CATEGORY "default")

if(COVISE_OPENCOVER_INTERNAL_PROJECT)
   set(COVISE_UTIL_LIBRARY coUtil)
   set(COVISE_FILE_LIBRARY coFile)
   set(COVISE_APPL_LIBRARY coAppl)
   set(COVISE_API_LIBRARY coApi)
   set(COVISE_CORE_LIBRARY coCore)
   set(COVISE_NET_LIBRARY coNet)
   set(COVISE_DO_LIBRARY coDo)
   set(COVISE_SHM_LIBRARY coShm)
   set(COVISE_CONFIG_LIBRARY coConfig)
   set(COVISE_VRBCLIENT_LIBRARY coVRBClient)
   set(COVISE_VRBCLIENT_LIBRARY coVRBServer)
   set(COVISE_GRMSG_LIBRARY coGRMsg)
   set(COVISE_GPU_LIBRARY coGPU)
   set(COVISE_ALG_LIBRARY coAlg)
   set(COVISE_IMAGE_LIBRARY coImage)
   set(COVISE_VTK_LIBRARY coVtk)

   set(COVISE_GPU_LIBRARIES ${COVISE_GPU_LIBRARY})

   set(COVISE_INCLUDE_DIRS ../kernel)
   set(COVISE_FOUND TRUE)
else()
   cmake_minimum_required(VERSION 3.1)
   getenv_path(COVISEDIR COVISEDIR)
   set(CMAKE_MODULE_PATH "${COVISEDIR}/cmake" ${CMAKE_MODULE_PATH})

   # global feature tests
   INCLUDE(CoviseGlobalFeatureTests)

   find_package(COVISE REQUIRED)
   find_package(COVISE COMPONENTS GPU REQUIRED)
   #covise_create_using()

   # preset some initial values for install dir and CFLAGS etc.
   INCLUDE(CovisePresets)
   covise_cmake_policy()

   include(CoviseHelperMacros)

   set(CMAKE_INSTALL_PREFIX "${COVISE_INSTALL_PREFIX}")

   INCLUDE(Qt5-6)
   find_qt()
endif()

if (POLICY CMP0020)
    #qt policy Automatically link Qt executables to qtmain target on Windows.
    cmake_policy( SET CMP0020 NEW)
endif()

IF(COVISE_BUILD_WEBSERVICE)
   COVISE_FIND_PACKAGE(GSOAP)
ENDIF(COVISE_BUILD_WEBSERVICE)

COVISE_FIND_PACKAGE(GLEW REQUIRED)
COVISE_FIND_PACKAGE(XERCESC REQUIRED)
set(OpenGL_GL_PREFERENCE LEGACY)
COVISE_FIND_PACKAGE(OpenGL REQUIRED)
COVISE_FIND_PACKAGE(Cg)
COVISE_FIND_PACKAGE(ARTOOLKIT)
COVISE_FIND_PACKAGE(ZLIB)
COVISE_FIND_PACKAGE(JPEGTURBO)
if (NOT JPEGTURBO_FOUND)
   COVISE_FIND_PACKAGE(JPEG)
endif()
COVISE_FIND_PACKAGE(PNG)
COVISE_FIND_PACKAGE(TIFF)
COVISE_FIND_PACKAGE(FFMPEG)
COVISE_FIND_PACKAGE(Threads)
COVISE_FIND_PACKAGE(VRPN)
IF(APPLE)
  FIND_LIBRARY(APP_SERVICES_LIBRARY ApplicationServices)
  MARK_AS_ADVANCED(APP_SERVICES_LIBRARY)
ENDIF(APPLE)

COVISE_FIND_CUDA()

covise_find_package(OpenSceneGraph 3.2.0 COMPONENTS osgDB osgUtil osgViewer
   osgParticle osgText osgSim osgGA osgFX osgTerrain osgShadow REQUIRED)

COVISE_FIND_PACKAGE(CURL QUIET)
IF(CURL_FOUND)
  ADD_DEFINITIONS(-DHAVE_LIBCURL)
  IF(WIN32)
    ADD_DEFINITIONS(-DCURL_STATICLIB)
  ENDIF(WIN32)
ENDIF()

if(NOT COVISE_OPENCOVER_INTERNAL_PROJECT)
    set(ARCHSUFFIX "${COVISE_ARCHSUFFIX}")
    string(TOLOWER "${ARCHSUFFIX}" ARCHSUFFIX)
    string(REGEX REPLACE "opt$" "" BASEARCHSUFFIX "${ARCHSUFFIX}")
    set(DBG_ARCHSUFFIX "${BASEARCHSUFFIX}")
    string(REGEX REPLACE "xenomai$" "" BASEARCHSUFFIX "${BASEARCHSUFFIX}")
    string(REGEX REPLACE "mpi$" "" BASEARCHSUFFIX "${BASEARCHSUFFIX}")
    if(ARCHSUFFIX STREQUAL DBG_ARCHSUFFIX)
        set(CMAKE_BUILD_TYPE "Debug")
    else()
        set(CMAKE_BUILD_TYPE "RelWithDebInfo")
    endif()
    message(CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()

if(COVISE_USE_X11)
    add_definitions(-DUSE_X11)
endif()

# find dependencies for coVRML library
using(FLEX optional)
covise_find_package(BISON)

include_directories(${COVISE_INCLUDE_DIRS})
include_directories(.)

if (COVISE_USE_OSMESA)
    add_definitions(-DUSE_OSMESA)
endif()



if(NOT COVISE_OPENCOVER_INTERNAL_PROJECT)
    IF(COVISE_EXPORT_TO_INSTALL)
        set(COVISE_EXPORTS_PATH "${COVISEDIR}/${ARCHSUFFIX}")
    ELSE()
        set(COVISE_EXPORTS_PATH "${CMAKE_BINARY_DIR}")
    ENDIF()
    FILE(WRITE ${COVISE_EXPORTS_PATH}/${COVISE_EXPORTS_FILE} "")
endif()

if (COVISE_BUILD_ONLY_COVER)
    set(COVISE_BUILD_DRIVINGSIM OFF)
else()
    add_definitions(-DUSE_COVISE)
endif()

ADD_SUBDIRECTORY(OpenConfig)

ADD_SUBDIRECTORY(cover/input/dev)
ADD_SUBDIRECTORY(DrivingSim)

USING(MPI optional)
USING(Pthreads optional)
using(Boost)

if (COVISE_USE_VIRVO)
   ADD_SUBDIRECTORY(osgcaveui)
endif()
ADD_SUBDIRECTORY(OpenVRUI)
ADD_SUBDIRECTORY(PluginUtil)
ADD_SUBDIRECTORY(cover)

ADD_SUBDIRECTORY(main)
ADD_SUBDIRECTORY(main.mpi)
ADD_SUBDIRECTORY(mpiwrapper)

ADD_SUBDIRECTORY(vrml97)
ADD_SUBDIRECTORY(plugins)
ADD_SUBDIRECTORY(ManagedOpenCOVER)
ADD_SUBDIRECTORY(Sur40COVER)
ADD_SUBDIRECTORY(convPagedLOD)

if(NOT COVISE_OPENCOVER_INTERNAL_PROJECT)
    IF(${CMAKE_VERSION} VERSION_GREATER "2.8.2")
        INCLUDE(FeatureSummary)
        FEATURE_SUMMARY(WHAT ALL)
    ENDIF()

    message("Building for ARCHSUFFIX: ${COVISE_ARCHSUFFIX}, installing to: ${COVISE_DESTDIR}")
endif()

INSTALL(FILES "../../cmake/FindCOVER.cmake" DESTINATION
    "${CMAKE_INSTALL_PREFIX}/share/cover/cmake/" RENAME "cover-config.cmake")

# IF(MSVC)
  # ADD_SUBDIRECTORY(ManagedOpenCOVER)
  # ADD_SUBDIRECTORY(SurfaceCOVER)
# ENDIF(MSVC)

# TODO: handle the following the "cmake way"
#
# QMAKE_EXTRA_TARGETS *= bundle opencoverapp

# bundle.depends = opencoverapp

# opencoverapp.depends = main plugins always_out_of_date
# opencoverapp.target = $$COVISEDIR/$$ARCHSUFFIX/Applications/OpenCOVER.app
# opencoverapp.commands = $(RM) -r "$@" \
    # && mkdir -p $$COVISEDIR/$$ARCHSUFFIX/Applications \
    # && cp -HR $$BINDESTDIR/Renderer/OpenCOVER.app "$@" \
    # && mkdir -p "$@"/Contents/Resources \
    # && cp -HR $$COVISEDIR/share/covise/icons $$COVISEDIR/share/covise/fonts $$COVISEDIR/config "$@"/Contents/Resources \
    # && mkdir -p "$@"/Contents/PlugIns \
    # && cp -HR ${OPENSCENEGRAPH_LIBPATH}/osgPlugins-2.9.7/* "$@"/Contents/PlugIns/ \
    # && cp -HR $$COVISEDIR/$$ARCHSUFFIX/lib/OpenCOVER/plugins/* "$@"/Contents/PlugIns/ \
    # && ${COVISEDIR}/../common/mkspecs/makebundle.sh "$@"
