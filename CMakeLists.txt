# @file
#
# Root-file for the Covise configure/build setup process
#
# @author Blasius Czink
#

# requirements to cmake version

# 2.8.7 is too old for identifying clang
CMAKE_MINIMUM_REQUIRED(VERSION 3.1...3.10)
if(APPLE)
  set (CMAKE_CXX_STANDARD 17) # I would like to use cpp 17 but OpenCV2 does not support it. so back to 11
else()
  set (CMAKE_CXX_STANDARD 17) # I would like to use cpp 17 but OpenCV2 does not support it. so back to 11
endif()
set (CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)

set(COVISE_BUILDING_COVISE TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# allow only out-of-source builds
STRING(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" in_source_build)
IF(in_source_build)
  MESSAGE(FATAL_ERROR "A separate build directory is required, please create one and run cmake from this directory")
ENDIF(in_source_build)

# covise project
PROJECT(covise VERSION 2019 LANGUAGES CXX C)

set(COVISE_EXPORTS_FILE "covise-exports.cmake")

########## General setup ##########
if(WIN32)
  set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
ENDIF()

# this is used to override some of cmake's original find-modules
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
IF(WIN32)
  SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/windows;${CMAKE_MODULE_PATH}")
ENDIF()
IF(UNIX)
  SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/unix;${CMAKE_MODULE_PATH}")
ENDIF()
IF(APPLE)
  SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/apple;${CMAKE_MODULE_PATH}")
ENDIF()
IF(MINGW)
  SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/mingw;${CMAKE_MODULE_PATH}")
ENDIF(MINGW)

message("CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
message("CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")

# set COVISEDIR, BASEARCHSUFFIX and some other cmake-variables
SET(COVISEDIR "${CMAKE_SOURCE_DIR}")
IF(NOT ARCHSUFFIX)
    SET(ARCHSUFFIX "$ENV{ARCHSUFFIX}")
ENDIF()
# archsuffix has to be set
IF(ARCHSUFFIX STREQUAL "")
   set(BITS "")
   if (NOT CMAKE_SIZEOF_VOID_P STREQUAL "")
       MATH(EXPR BITS ${CMAKE_SIZEOF_VOID_P}*8)
   endif()
   SET(ARCHSUFFIX ${CMAKE_SYSTEM_NAME}${BITS}opt)
   STRING(TOLOWER "${ARCHSUFFIX}" ARCHSUFFIX)
   MESSAGE(WARNING "Environment variable ARCHSUFFIX was not set, using ARCHSUFFIX=${ARCHSUFFIX}")
ENDIF()

STRING(TOLOWER "${ARCHSUFFIX}" ARCHSUFFIX)
STRING(REGEX REPLACE "opt$" "" BASEARCHSUFFIX "${ARCHSUFFIX}")
SET(DBG_ARCHSUFFIX "${BASEARCHSUFFIX}")
STRING(REGEX REPLACE "xenomai$" "" BASEARCHSUFFIX "${BASEARCHSUFFIX}")
STRING(REGEX REPLACE "mpi$" "" BASEARCHSUFFIX "${BASEARCHSUFFIX}")
IF(NOT WIN32 AND "" STREQUAL "${CMAKE_BUILD_TYPE}")
  IF(ARCHSUFFIX STREQUAL DBG_ARCHSUFFIX)
    SET(CMAKE_BUILD_TYPE "Debug")
  ELSE()
    SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
  ENDIF()
ENDIF()
MESSAGE(CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
set(COVISE_ARCHSUFFIX ${ARCHSUFFIX})

IF(UNIX AND NOT APPLE)
  EXECUTE_PROCESS(COMMAND readlink -e ${COVISEDIR}
      OUTPUT_VARIABLE CANONICAL_COVISE_DIR
      ERROR_VARIABLE CANONICAL_COVISE_DIR_error
      RESULT_VARIABLE CANONICAL_COVISE_DIR_result
      OUTPUT_STRIP_TRAILING_WHITESPACE)
ELSE()
  SET(CANONICAL_COVISE_DIR ${COVISEDIR})
ENDIF()

if (NOT BASEARCHSUFFIX STREQUAL "spack")
    if (APPLE AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
        set(CMAKE_PREFIX_PATH /opt/homebrew ${CMAKE_PREFIX_PATH})
        set(CMAKE_FRAMEWORK_PATH /opt/homebrew/Frameworks ${CMAKE_FRAMEWORK_PATH})
    endif()
endif()

# figure out COVISE version
FIND_PACKAGE(Git)

set(COVISE_VERSION_YEAR 2015)
set(COVISE_VERSION_MONTH 1)
set(COVISE_VERSION_DAY 0)
set(COVISE_GIT_DATE 1970-01-01)
set(COVISE_VERSION_HASH "")
if (GIT_FOUND)
   execute_process(COMMAND ${GIT_EXECUTABLE} log -n1 --format=%h/%ci
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE GIT_LOG_OUT
      ERROR_VARIABLE GIT_LOG_ERROR
      RESULT_VARIABLE GIT_LOG_RESULT
      OUTPUT_STRIP_TRAILING_WHITESPACE)

   if(NOT ${GIT_LOG_RESULT} EQUAL 0)
      message("Command \"${GIT_EXECUTABLE} log -n1 --format=%H/%ci ${dir}\" failed with output:\n${GIT_LOG_ERROR}")
      set(COVISE_VERSION "2015.1")
      set(COVISE_VERSION_HASH "unknown")
   else()
      string(REGEX REPLACE "^([a-z0-9]*)/(....)-(..)-(..).*" "\\2-\\3-\\4" COVISE_GIT_DATE "${GIT_LOG_OUT}")
      string(REGEX REPLACE "^([a-z0-9]*)/(....)-(..)-(..).*" "\\1" COVISE_VERSION_HASH "${GIT_LOG_OUT}")
      string(REGEX REPLACE "^([a-z0-9]*)/(....)-(..)-(..).*" "\\2" COVISE_VERSION_YEAR "${GIT_LOG_OUT}")
      string(REGEX REPLACE "^([a-z0-9]*)/(....)-(..)-(..).*" "\\3" COVISE_VERSION_MONTH "${GIT_LOG_OUT}")
      string(REGEX REPLACE "^0([0-9]*)" "\\1" COVISE_VERSION_MONTH "${COVISE_VERSION_MONTH}")
      string(REGEX REPLACE "^([a-z0-9]*)/(....)-(..)-(..).*" "\\4" COVISE_VERSION_DAY "${GIT_LOG_OUT}")
      string(REGEX REPLACE "^0([0-9]*)" "\\1" COVISE_VERSION_DAY "${COVISE_VERSION_DAY}")
      #message("git log output: ${GIT_LOG_OUT}")
   endif()
endif()
SET(COVISE_MAJOR_VERSION "${COVISE_VERSION_YEAR}")
SET(COVISE_MINOR_VERSION "${COVISE_VERSION_MONTH}")
SET(COVISE_PATCH_VERSION "${COVISE_VERSION_DAY}")
SET(COVISE_VERSION "${COVISE_VERSION_YEAR}.${COVISE_VERSION_MONTH}-${COVISE_VERSION_HASH}")

MESSAGE("COVISEDIR=${COVISEDIR}")
MESSAGE("ARCHSUFFIX=${ARCHSUFFIX}, BASEARCHSUFFIX=${BASEARCHSUFFIX}")
MESSAGE("COVISE_VERSION = ${COVISE_VERSION}, Last git change on ${COVISE_GIT_DATE}")

# CLANG-TIDY option
OPTION(BUILD_USE_CLANG_TIDY "use clang-tidy framework for diagnosing and fixing typical programming errors" OFF)
MARK_AS_ADVANCED(BUILD_USE_CLANG_TIDY)
if (BUILD_USE_CLANG_TIDY)
    SET(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,boost-*,bugprone-*,cert-*,clang-analyzer-*,misc-*,performance-*,readability-*,cppcoreguidelines-*)
    #SET(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,boost-*,bugprone-*,cert-*,clang-analyzer-*,misc-*,modernize-*,performance-*,readability-redundant-*,readability-string-compare,google-readability-casting,cppcoreguidelines-avoid-goto,readability-function-size,-modernize-use-trailing-return-type)
    MARK_AS_ADVANCED(CMAKE_CXX_CLANG_TIDY)
else()
    UNSET(CMAKE_CXX_CLANG_TIDY)
endif()

# COVISE global options
option(COVISE_CMAKE_VERBOSE "more verbose CMake output" OFF)


OPTION(COVISE_BUILD_ONLY_FILE "only build COVISE file I/O library" OFF)
OPTION(COVISE_BUILD_ONLY_COVER "only build OpenCOVER" OFF)
OPTION(COVISE_BUILD_ONLY_ODDLOT "only build OddLOT" OFF)

set(COVISE_BUILD_ONLY_COMPONENTS ${COVISE_BUILD_ONLY_COMPONENTS} COVISE_BUILD_ONLY_FILE COVISE_BUILD_ONLY_COVER COVISE_BUILD_ONLY_ODDLOT)
set(NUM_COVISE_BUILD_ONLY_COMPONENTS 0)
foreach(COVISE_BUILD_ONLY_VAR ${COVISE_BUILD_ONLY_COMPONENTS})
  if(${COVISE_BUILD_ONLY_VAR})
    MATH(EXPR NUM_COVISE_BUILD_ONLY_COMPONENTS "${NUM_COVISE_BUILD_ONLY_COMPONENTS}+1")
  endif()
endforeach()

if(${NUM_COVISE_BUILD_ONLY_COMPONENTS} GREATER 1)
  message(FATAL_ERROR "COVISE_BUILD_ONLY_... options are mutually exclusive")
endif()


option(COVISE_OPENCOVER_INTERNAL_PROJECT "build OpenCOVER from this project" ON)
if (BASEARCHSUFFIX STREQUAL "cray64")
    OPTION(COVISE_USE_OSMESA "Use OSMesa OpenGL library for off-screen CPU rendering" ON)
else()
    OPTION(COVISE_USE_OSMESA "Use OSMesa OpenGL library for off-screen CPU rendering" OFF)
endif()
OPTION(COVISE_USE_FOLDERS "Enable solution folders in Visual Studio. Disable for Express versions." ON)
OPTION(COVISE_EXPORT_TO_INSTALL "Place COVISE exported targets cmake-file in <install directory>/<archsuffix>/lib (default). Otherwise in build directory." ON)
if(APPLE)
    OPTION(COVISE_USE_QT5 "Use Qt 5" ON)
else()
    OPTION(COVISE_USE_QT5 "Use Qt 5" ON)
endif()

if(DEFINED  $ENV{COVISE_USE_QT5})
  set(COVISE_USE_QT5 $ENV{COVISE_USE_QT5})
endif()
OPTION(COVISE_WARNING_IS_ERROR "Treat warnings as errors" OFF)
if (CMAKE_GENERATOR STREQUAL "Ninja" OR CMAKE_GENERATOR STREQUAL "Ninja Multi-Config")
  OPTION(COVISE_USE_FORTRAN "Enable Modules using Fortran" OFF)
  OPTION(COVISE_USE_CSHARP "Enable C#" OFF)
else()
  OPTION(COVISE_USE_FORTRAN "Enable Modules using Fortran" ON)
  OPTION(COVISE_USE_CSHARP "Enable C#" OFF)
endif()

set(COVISE_CPU_ARCH "native" CACHE STRING "CPU architecture to optimize for (native, core-avx2, atom, ...)")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60) # be compatible with P6000/GP100
endif()

option(COVISE_SANITIZE_UNDEFINED "Build with -fsanitize=undefined" OFF)
option(COVISE_SANITIZE_ADDRESS "Build with -fsanitize=address" OFF)
option(COVISE_SANITIZE_THREAD "Build with -fsanitize=thread" OFF)
option(COVISE_IGNORE_RETURNED "No compile error when return values are ignored" OFF)

OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)

OPTION(COVISE_BUILD_SYS "Build COVISE system applications" ON)
OPTION(COVISE_BUILD_MODULES "Build COVISE modules" ON)

OPTION(COVISE_BUILD_WEBSERVICE "Build web service interface" OFF)
OPTION(COVISE_BUILD_DRIVINGSIM "Build driving simulator" ON)

OPTION(COVISE_USE_CUDA "Use CUDA" ON)
OPTION(COVISE_USE_MPI "Use MPI" ON)

if(WIN32 OR APPLE OR COVISE_USE_OSMESA)
    OPTION(COVISE_USE_X11 "Use X11" OFF)
else()
    OPTION(COVISE_USE_X11 "Use X11" ON)
endif()

OPTION(COVISE_USE_OPENCV3 "Use OpenCV3" OFF)
OPTION(COVISE_USE_OPENCV4 "Use OpenCV4" ON)
OPTION(COVISE_USE_MERCURY "Use MERCURY" ON)

if (IS_DIRECTORY "${PROJECT_SOURCE_DIR}/src/3rdparty/visionaray")
    OPTION(COVISE_USE_VISIONARAY "Use Visionaray raytracing library" OFF)
else()
    OPTION(COVISE_USE_VISIONARAY "Use Visionaray raytracing library" OFF)
endif()

if (IS_DIRECTORY "${PROJECT_SOURCE_DIR}/src/3rdparty/deskvox")
    OPTION(COVISE_USE_VIRVO "Use Virvo volume rendering library" ON)
else()
    OPTION(COVISE_USE_VIRVO "Use Virvo volume rendering library" OFF)
    message("Did not find src/3rdparty/deskvox submodule: disabling direct volume rendering")
    message("Maybe run \"git submodule update --init --recursive\"?")
endif()

OPTION(COVISE_BUILD_COVER "Build OpenCOVER VR/desktop renderer" ON)
OPTION(COVISE_USE_VISIBILITY "Do not export all symbols" ON)

IF(COVISE_USE_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
ENDIF(COVISE_USE_FOLDERS)
# enable "Fortran" (keep it optional and check availability with CMAKE_Fortran_COMPILER_WORKS where needed)
IF(COVISE_USE_FORTRAN)
  ENABLE_LANGUAGE(Fortran OPTIONAL)
ENDIF(COVISE_USE_FORTRAN)

SET(CPACK_GENERATOR "TGZ") 
SET(CPACK_PACKAGE_FILE_NAME covise-${ARCHSUFFIX})
INCLUDE(CPackConfig)
INCLUDE(CPack)

# COVISE helper macros 
INCLUDE(Qt5-6)
# preset some initial values for install dir and CFLAGS etc.
INCLUDE(CovisePresets)
covise_cmake_policy()
if (POLICY CMP0074)
  # make find_include/find_library search in <PackageName>_ROOT prefix
  cmake_policy(SET CMP0074 NEW)
endif()

INCLUDE(CoviseHelperMacros)
INCLUDE(ConfigureLinker)

# global feature tests
INCLUDE(CoviseGlobalFeatureTests)

IF(COVISE_EXPORT_TO_INSTALL)
   set(COVISE_EXPORTS_PATH "${COVISEDIR}/${ARCHSUFFIX}")
ELSE()
   set(COVISE_EXPORTS_PATH "${CMAKE_BINARY_DIR}")
ENDIF()
FILE(WRITE ${COVISE_EXPORTS_PATH}/${COVISE_EXPORTS_FILE} "")

# Create macro list for USING
CREATE_USING()
if(WIN32)
  OPTION(COVISE_BOOST_STATIC "set Boost_USE_STATIC_LIBS" OFF)
  set(Boost_USE_STATIC_LIBS ${COVISE_BOOST_STATIC})
endif()
COVISE_FIND_BOOST()

if (NOT COVISE_BUILD_ONLY_FILE)
    # find important covise dependencies
    IF(APPLE)
        FIND_LIBRARY(APP_SERVICES_LIBRARY ApplicationServices)
        MARK_AS_ADVANCED(APP_SERVICES_LIBRARY)
    ENDIF(APPLE)
    IF(COVISE_BUILD_WEBSERVICE)
        COVISE_FIND_PACKAGE(GSOAP)
    ENDIF(COVISE_BUILD_WEBSERVICE)
    find_qt()
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    COVISE_FIND_PACKAGE(Threads)
    COVISE_FIND_PACKAGE(PTHREAD)
    #COVISE_FIND_PACKAGE(Pthreads)
    COVISE_FIND_PACKAGE(OpenSSL)
    set(OpenGL_GL_PREFERENCE LEGACY)
    COVISE_FIND_PACKAGE(OpenGL)
    COVISE_FIND_PACKAGE(GLEW) # virvo depends on it
    if (NOT GLEW_FOUND)
        set (COVISE_USE_VIRVO OFF)
        message("Not building Virvo: no GLEW")
    endif()
    if (COVISE_USE_VIRVO)
        covise_find_package(GDCM QUIET)
    endif()
    COVISE_FIND_PACKAGE(XERCESC REQUIRED)
    COVISE_FIND_PACKAGE(ZLIB REQUIRED)
    COVISE_FIND_PACKAGE(JPEGTURBO)
    if (NOT JPEGTURBO_FOUND)
        COVISE_FIND_PACKAGE(JPEG REQUIRED)
    endif()
    COVISE_FIND_PACKAGE(PNG REQUIRED)
    COVISE_FIND_PACKAGE(TIFF REQUIRED)
    if (NOT COVISE_BUILD_ONLY_ODDLOT)
        COVISE_FIND_PACKAGE(GLUT)
        COVISE_FIND_PACKAGE(FFMPEG)
        COVISE_FIND_PACKAGE(Steereo)
        covise_find_package(OpenSceneGraph 3.2.0 COMPONENTS osgDB osgUtil osgViewer osgParticle osgText osgSim osgGA osgFX osgTerrain)
        covise_find_package(VTK COMPONENTS vtkIOLegacy vtkIOXML vtkFiltersCore vtkCommonCore vtkImagingCore vtkCommonDataModel vtkCommonExecutionModel NO_MODULE)
        covise_find_package(HDF5 COMPONENTS C HL CXX)

        COVISE_FIND_PACKAGE(Cg)

        covise_find_cuda()

        COVISE_FIND_PACKAGE(PythonLibs)
        covise_find_package(OpenCV)
    endif(NOT COVISE_BUILD_ONLY_ODDLOT)
    covise_find_package(OPENCRG)
endif()

if (APPLE)
    add_definitions(-DGL_SILENCE_DEPRECATION)
endif()
if(MSVC)
    add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()
# descend to sources
ADD_SUBDIRECTORY(src)

OPTION(COVISE_BUILD_TESTS "compile the tests" OFF)
if(COVISE_BUILD_TESTS)
ADD_SUBDIRECTORY(tests)
endif()

# Install

SET(install_files_exclude "(/CVS/|/\\.svn/|/\\.hg/|/\\.git/|\\.swp$|\\.#|/#|/\\.gitignore|/\\.hgignore|\\.html\\.in$|\\.tex\\.in$|\\.tex$")
IF(UNIX)
  SET(install_files_exclude "${install_files_exclude}|\\.bat$|\\.vbs$")
ENDIF(UNIX)
SET(install_files_exclude "${install_files_exclude}|bin/sasuctrl")
if(COVISE_BUILD_ONLY_COVER)
    SET(install_files_exclude "${install_files_exclude}|share/covise/example-data|share/covise/signals|share/covise/web")
endif()
SET(install_files_exclude "${install_files_exclude})")

#CONFIGURE_FILE("${COVISEDIR}/cmake/FindCovise.cmake" "${CMAKE_BINARY_DIR}/cmake/FindCovise.cmake" COPYONLY)
#CONFIGURE_FILE("${COVISEDIR}/cmake/FindZLIB.cmake" "${CMAKE_BINARY_DIR}/cmake/FindZLIB.cmake" COPYONLY)
#CONFIGURE_FILE("${COVISEDIR}/cmake/CoviseHelperMacros.cmake" "${CMAKE_BINARY_DIR}/cmake/CoviseHelperMacros.cmake" COPYONLY)
#INSTALL(DIRECTORY "${CMAKE_BINARY_DIR}/cmake" DESTINATION "${CMAKE_INSTALL_PREFIX}/share/covise/cmake")
FILE(GLOB cmake_files "cmake/*.cmake")
FILE(GLOB using_files "cmake/Using/*")
INSTALL(FILES ${using_files} DESTINATION "${CMAKE_INSTALL_PREFIX}/share/covise/cmake/Using")
INSTALL(FILES "cmake/FindCOVISE.cmake" DESTINATION
    "${CMAKE_INSTALL_PREFIX}/share/covise/cmake/" RENAME "covise-config.cmake")

INSTALL(FILES .covise .covise.sh
              DESTINATION .)

INSTALL(FILES
    config/config.xml
    config/config-colormaps.xml
    config/config-filetypes.xml
    config/config-ar.xml
    config/config-spray.xml
    config/config-midi.xml
    DESTINATION config)

INSTALL(DIRECTORY bin scripts
        USE_SOURCE_PERMISSIONS
        DESTINATION .
        REGEX "${install_files_exclude}" EXCLUDE)

INSTALL(DIRECTORY share
        USE_SOURCE_PERMISSIONS
        DESTINATION .
        REGEX "${install_files_exclude}" EXCLUDE)

INSTALL(FILES
    README.md README-3rd-party.txt README-ARCHSUFFIX.txt
    DESTINATION share/doc/covise)
#INSTALL(DIRECTORY doc/pdf doc/html
#   DESTINATION share/doc/covise)

# in order to have an "uninstall" target
CONFIGURE_FILE(
  "${COVISEDIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

configure_file(
  "${COVISEDIR}/cmake/CoviseOptions.cmake.in"
  "${COVISE_EXPORTS_PATH}/CoviseOptions.cmake" @ONLY)
install(FILES
    "${COVISE_EXPORTS_PATH}/CoviseOptions.cmake"
    "${COVISE_EXPORTS_PATH}/CoviseUsingMacros.cmake"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/${ARCHSUFFIX}")
#INSTALL(FILES "${COVISE_EXPORTS_PATH}/CoviseUsingMacros.cmake" ${cmake_files} DESTINATION "${CMAKE_INSTALL_PREFIX}/share/covise/cmake")
INSTALL(FILES ${cmake_files} DESTINATION "${CMAKE_INSTALL_PREFIX}/share/covise/cmake")

ADD_CUSTOM_TARGET(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")

IF(${CMAKE_VERSION} VERSION_GREATER "2.8.2")
  INCLUDE(FeatureSummary)
  FEATURE_SUMMARY(WHAT ALL)
ENDIF()
