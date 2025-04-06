# @file
#
# CMakeLists.txt for covise renderer VIVE
#
# @author Uwe Woessner


PROJECT(VIVE C CXX)

set(CMAKE_PREFIX_PATH $ENV{VK_SDK_PATH} ${CMAKE_PREFIX_PATH})

USING(GDAL)

USING(vsg)


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

set(COVISE_INCLUDE_DIRS ../core)
set(COVISE_FOUND TRUE)

if (POLICY CMP0020)
    #qt policy Automatically link Qt executables to qtmain target on Windows.
    cmake_policy( SET CMP0020 NEW)
endif()




if(COVISE_USE_QT)
    add_definitions(-DUSE_QT)
endif()

if(COVISE_USE_X11)
    add_definitions(-DUSE_X11)
endif()

# find dependencies for vvVRML library
using(FLEX optional)
covise_find_package(BISON)
COVISE_FIND_PACKAGE(CURL)
using(CURL optional)

include_directories(${COVISE_INCLUDE_DIRS})
include_directories(.)


ADD_SUBDIRECTORY(OpenConfig)

ADD_SUBDIRECTORY(core)
ADD_SUBDIRECTORY(main)
ADD_SUBDIRECTORY(main.mpi)
ADD_SUBDIRECTORY(mpiwrapper)
ADD_SUBDIRECTORY(core/input/dev)

#ADD_SUBDIRECTORY(vrml97)

