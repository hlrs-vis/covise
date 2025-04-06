# @file
#
# CMakeLists.txt for covise renderer OpenCOVER
#
# @author Blasius Czink

#COVISE_UNFINISHED()

PROJECT(OpenCOVER C CXX)
# this is already set in the top most CMakeLists.txt set (CMAKE_CXX_STANDARD 11)
set(COVISE_BUILDING_COVER TRUE)

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
set(COVISE_GRMSG_LIBRARY coGRMsg)
set(COVISE_GPU_LIBRARY coGPU)
set(COVISE_ALG_LIBRARY coAlg)
set(COVISE_IMAGE_LIBRARY coImage)
set(COVISE_VTK_LIBRARY coVtk)

set(COVISE_GPU_LIBRARIES ${COVISE_GPU_LIBRARY})

set(COVISE_INCLUDE_DIRS ../kernel)
set(COVISE_FOUND TRUE)

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
if(NOT TARGET OpenGL::GLU)
    message("OpenCOVER requires GLU")
    return()
endif()
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

COVISE_FIND_PACKAGE(CURL)

if(COVISE_USE_QT)
    add_definitions(-DUSE_QT)
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
using(CURL optional)

if (COVISE_USE_VIRVO)
   ADD_SUBDIRECTORY(osgcaveui)
endif()
ADD_SUBDIRECTORY(OpenVRUI)
ADD_SUBDIRECTORY(PluginUtil)
ADD_SUBDIRECTORY(OpcUaClient)
ADD_SUBDIRECTORY(HTTPClient)
ADD_SUBDIRECTORY(cover)
ADD_SUBDIRECTORY(main)
ADD_SUBDIRECTORY(main.mpi)
ADD_SUBDIRECTORY(mpiwrapper)

ADD_SUBDIRECTORY(vrml97)
ADD_SUBDIRECTORY(convPagedLOD)
ADD_SUBDIRECTORY(createPagedLOD)
ADD_SUBDIRECTORY(tileCitygml)
ADD_SUBDIRECTORY(fixLOD)
ADD_SUBDIRECTORY(plugins)
ADD_SUBDIRECTORY(ManagedOpenCOVER)
ADD_SUBDIRECTORY(Sur40COVER)
ADD_SUBDIRECTORY(utils)

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
# opencoverapp.target = $$COVISEDIR/$$COVISE_ARCHSUFFIX/Applications/OpenCOVER.app
# opencoverapp.commands = $(RM) -r "$@" \
# && mkdir -p $$COVISEDIR/$$COVISE_ARCHSUFFIX/Applications \
    # && cp -HR $$BINDESTDIR/Renderer/OpenCOVER.app "$@" \
    # && mkdir -p "$@"/Contents/Resources \
    # && cp -HR $$COVISEDIR/share/covise/icons $$COVISEDIR/share/covise/fonts $$COVISEDIR/config "$@"/Contents/Resources \
    # && mkdir -p "$@"/Contents/PlugIns \
    # && cp -HR ${OPENSCENEGRAPH_LIBPATH}/osgPlugins-2.9.7/* "$@"/Contents/PlugIns/ \
    # && cp -HR $$COVISEDIR/$$COVISE_ARCHSUFFIX/lib/OpenCOVER/plugins/* "$@"/Contents/PlugIns/ \
    # && ${COVISEDIR}/../common/mkspecs/makebundle.sh "$@"
