# @file CovisePresets.cmake
#
# Initialise install dir, covise specific CFLAGS etc. here
#
# @author Blasius Czink

# for the lazy ones who do not want to type the same thing in else(), endif() and so on
SET(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)

if ("${COVISE_ARCHSUFFIX}" MATCHES "^spack")
    set(COVISE_INSTALL_LIBDIR lib)
else()
    set(COVISE_INSTALL_LIBDIR ${COVISE_ARCHSUFFIX}/lib)
endif()
# in order to allow linking to libraries installed with spack
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_MACOSX_RPATH TRUE)

macro(covise_cmake_policy)
# policy settings
IF(COMMAND cmake_policy)
  
    IF(POLICY CMP0077)
       cmake_policy(SET CMP0077 NEW)
    ENDIF()
	
    IF(POLICY CMP0020)
       #qt policy Automatically link Qt executables to qtmain target on Windows.
       cmake_policy(SET CMP0020 NEW)
    ENDIF()
	
	# allow LOCATION to be used in build-targets we might change to $<TARGET_FILE> if we need a newer CMAKE version
    #if (POLICY CMP0026)
    #    cmake_policy(SET CMP0026 OLD) # needed on windows
    #endif()
	
    
    # Works around warnings libraries linked against that don't have absolute paths (e.g. -lpthreads)
    cmake_policy(SET CMP0003 NEW)

    if(POLICY CMP0060)
        cmake_policy(SET CMP0060 NEW) # always link to full path never convert to -lLibraryName
    endif()
    # Works around warnings about escaped quotes in ADD_DEFINITIONS statements.

    if(POLICY CMP0071)
        cmake_policy(SET CMP0071 NEW) # always link to full path never convert to -lLibraryName
    endif()

    cmake_policy(SET CMP0005 NEW)

    # allow the commands include() and COVISE_FIND_PACKAGE() to do their default cmake_policy PUSH and POP.
    cmake_policy(SET CMP0011 NEW)

    # do not require BUNDLE DESTINATION on MacOS application bundles (simply use RUNTIME DESTINATION)
    cmake_policy(SET CMP0006 NEW)

    if(POLICY CMP0042)
       # default to finding shlibs relative to @rpath on MacOS
       cmake_policy(SET CMP0042 NEW)
    endif()

    if(POLICY CMP0043)
       # configuration (RelWithDebInfo, Debug, ...) dependent COMPILE_DEFINITIONS are not used
       # - default to new behavior
       cmake_policy(SET CMP0043 NEW)
    endif()

    if (POLICY CMP0048)
        cmake_policy(SET CMP0048 NEW)
    endif()
	
    if(POLICY CMP0054)
       # in if()'s, only deref unquoted variable names
       cmake_policy(SET CMP0054 NEW)
    endif()

    if (POLICY CMP0074)
        # make find_include/find_library search in <PackageName>_ROOT prefix
        cmake_policy(SET CMP0074 NEW)
    endif()
	
    if (POLICY CMP0086)
        cmake_policy(SET CMP0086 NEW)
    endif()
	
    if (POLICY CMP0111)
        cmake_policy(SET CMP0111 NEW)
    endif()

    if (POLICY CMP0148)
        # for compatibility with open62541: still uses FindPythonInterp
        cmake_policy(SET CMP0148 OLD)
    endif()
	
ENDIF()
endmacro(covise_cmake_policy)


set(CUDA_PROPAGATE_HOST_FLAGS OFF) # not working on windows and probably not necessary on linux

IF(WIN32)
  # reenable the following if you want "d" suffix on debug versions of libraries and executables
  # please note that source and script changes are necessary in COVISE to make it run with suffixed executables/plugins
  # IF(MSVC)
  #   # our debug-suffix (only for VS)
  #   SET(CMAKE_DEBUG_POSTFIX  "d")
  # ENDIF()
ENDIF()

if(WIN32)
    set(BOOST_ROOT "$ENV{EXTERNLIBS}/boost")
    STRING(REGEX REPLACE "\\\\" "/" BOOST_ROOT ${BOOST_ROOT}) 
    #set(MPI_HOME "$ENV{EXTERNLIBS}/OpenMPI")
    #add_definitions("-DOMPI_IMPORTS")
    add_definitions("-DWIN32_LEAN_AND_MEAN")
    add_definitions("-DNOMINMAX")
    add_definitions("-DBOOST_ALL_NO_LIB")
    #add_definitions("-DBOOST_ALL_DYN_LINK")
endif(WIN32)

# do not reconfigure when doing a rebuild/clean for example in VS
# if you wish to reconfigure use cmake, ccmake or cmakesetup directly
#SET(CMAKE_SUPPRESS_REGENERATION false)

set(COVISE_DESTDIR ${COVISEDIR})
if(NOT "$ENV{COVISEDESTDIR}" STREQUAL "" AND NOT COVISE_DESTDIR STREQUAL "$ENV{COVISEDESTDIR}")
    message("COVISE internal build: COVISE_DESTDIR reset from $ENV{COVISEDESTDIR} to ${COVISE_DESTDIR}")
endif()

# preset install dir
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX "/usr/local/covise" CACHE PATH "CMAKE_INSTALL_PREFIX" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(ASAN_COMPILE_FLAGS "")
set(ASAN_LINK_FLAGS "")
if (COVISE_SANITIZE_UNDEFINED)
   set(ASAN_LINK_FLAGS "${ASAN_LINK_FLAGS} -fsanitize=undefined")
   set(ASAN_COMPILE_FLAGS "${ASAN_COMPILE_FLAGS} -fsanitize=undefined")
endif()
if (COVISE_SANITIZE_ADDRESS)
   set(ASAN_LINK_FLAGS "${ASAN_LINK_FLAGS} -fsanitize=address")
   set(ASAN_COMPILE_FLAGS "${ASAN_COMPILE_FLAGS} -fsanitize=address")
   if(WIN32)
        add_compile_options(/fsanitize=address)
    endif()
else()
endif()
if (COVISE_SANITIZE_THREAD)
   set(ASAN_LINK_FLAGS "${ASAN_LINK_FLAGS} -fsanitize=thread")
   set(ASAN_COMPILE_FLAGS "${ASAN_COMPILE_FLAGS} -fsanitize=thread")
endif()
if (COVISE_SANITIZE_THREAD OR COVISE_SANITIZE_ADDRESS OR COVISE_SANITIZE_UNDEFINED)
   set(ASAN_COMPILE_FLAGS "${ASAN_COMPILE_FLAGS} -fno-omit-frame-pointer")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_COMPILE_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${ASAN_LINK_FLAGS}")

IF(NOT WIN32 AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    if(NOT COVISE_CPU_ARCH STREQUAL "")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=${COVISE_CPU_ARCH}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${COVISE_CPU_ARCH}")
    endif()
ENDIF(NOT WIN32 AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")

if(COVISE_IGNORE_RETURNED)
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=unused-result")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-result")
endif()

if(BUILD_SHARED_LIBS)
   set(COVISE_LIB_TYPE SHARED)
else()
   set(COVISE_LIB_TYPE STATIC)
endif()

# change initial cmake-values such as CMAKE_CXX_FLAGS
# this is done only once at the first configure run
IF(NOT COVISE_CONFIGURED_ONCE)
  # Change default values here...
  # For example modify initial CXXFLAGS, CFLAGS for a specific compiler / archsuffix ...
  
  IF(APPLE)
    IF(BASEARCHSUFFIX STREQUAL "leopard")
      SET(CMAKE_OSX_ARCHITECTURES "x86_64;i386" CACHE STRING "Build architectures for OSX" FORCE)
    ENDIF()
  ENDIF(APPLE)

  IF(COVISE_USE_FORTRAN)
    # continue to support old ortran code with gnu fortran 8
    SET (CMAKE_Fortran_FLAGS "-std=legacy" CACHE STRING "general fortran flags" FORCE)
  ENDIF()

  # IF(CMAKE_COMPILER_IS_GNUCXX)
  #   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -fno-omit-frame-pointer")
  #   SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-strict-aliasing -fexceptions -fno-omit-frame-pointer")
  # ENDIF()

  # Then force modified default value(s) back into cache
  #
  # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "Flags used by the compiler during all build types" FORCE)
  # SET(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "Flags used by the compiler during all build types" FORCE)

  IF(CMAKE_COMPILER_IS_GNUCC)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99" CACHE STRING "Flags used by the compiler during all build types" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined"
       CACHE STRING "Flags used by the linker for shared objects" FORCE)
  ENDIF()
  
  # Defaults fixed on the first configure run! Avoid fixing values again!
  SET(COVISE_CONFIGURED_ONCE 1 CACHE INTERNAL "Covise was configured at least once.")
ENDIF(NOT COVISE_CONFIGURED_ONCE)

# make sure other external projects (like virvo) have an easy way to detect a covise cmake build run
SET(COVISE_BUILD ON)

# add initial per target compile flags to COVISE_COMPILE_FLAGS variable
# and initial per target link flags to COVISE_LINK_FLAGS
# the contents of these variables is used by ADD_COVISE_LIBRARY() and ADD_COVISE_EXECUTABLE() macros
# to add target specific compile and link flags
SET(COVISE_COMPILE_FLAGS "")
SET(COVISE_LINK_FLAGS    "")

# COVISE_WERROR_FLAG should contain the correct "warnings-are-errors" flag for the used compiler
IF(MSVC)
  SET(COVISE_WERROR_FLAG "-WX")
ELSE()
  SET(COVISE_WERROR_FLAG "-Werror")
ENDIF()

# treat warnings as errors?
IF(COVISE_WARNING_IS_ERROR)
    set(CMAKE_COMPILE_WARNING_AS_ERROR TRUE)
ELSE()
    set(CMAKE_COMPILE_WARNING_AS_ERROR FALSE)
  # we are assuming that most compilers do not treat warnings as errors if no special flag is set
  # if we are wrong place check and flag setting here
ENDIF()

# add more flags to COVISE_COMPILE_FLAGS or COVISE_LINK_FLAGS here if you wish
# Example: SET(COVISE_COMPILE_FLAGS "${COVISE_COMPILE_FLAGS} -fno-strict-aliasing")
# later in the CMakeLists.txt you should use:
# ADD_COVISE_COMPILE_FLAGS() or REMOVE_COVISE_COMPILE_FLAGS()
# and ADD_COVISE_LINK_FLAGS() or REMOVE_COVISE_LINK_FLAGS()
# to modify flags on a per target basis

# get rid of annoying template needs to have dll-interface warnings on VisualStudio
IF(MSVC)
  SET(COVISE_COMPILE_FLAGS "${COVISE_COMPILE_FLAGS} -wd4251 -wd4335")
ENDIF(MSVC)

# global defines
IF(WIN32)
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS -DFD_SETSIZE=512 -DXP_WIN -D_BOOL -D_USE_MATH_DEFINES)
  REMOVE_DEFINITIONS(-DUNICODE)
ENDIF(WIN32)

IF(MINGW)
   ADD_DEFINITIONS(-DSTRSAFE_NO_DEPRECATE)
ENDIF()

if(APPLE)
   if(BASEARCHSUFFIX STREQUAL "libc++")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
   endif()
endif(APPLE)

# include directories we need almost everywhere
INCLUDE_DIRECTORIES(
  "${COVISEDIR}/src/kernel"
)

# make sure ${CMAKE_CURRENT_BINARY_DIR} and ${CMAKE_CURRENT_SOURCE_DIR} are added automatically
SET(CMAKE_INCLUDE_CURRENT_DIR ON)
