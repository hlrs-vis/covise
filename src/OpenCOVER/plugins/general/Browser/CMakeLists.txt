#OpenCOVER plugins general Browser based on Chromium Embedded Framework
if(NOT WIN32 AND NOT APPLE)
  if(DEFINED  ENV{EXTERNLIBS})
    set(CEF_VERSION CACHE STRING "if set, build cef and install it in extern_libs")
    if(NOT ${CEF_VERSION} STREQUAL "")
      #set to newer version if required
      if(NOT EXISTS $ENV{EXTERNLIBS}/cef/cef_binary_${CEF_VERSION}_linux64.tar.bz2)
        message("building chromium embedded framework version " ${CEF_VERSION})
        #warn on overriding old build
        if(EXISTS $ENV{EXTERNLIBS}/ALL/libcef.so)
          message(WARNING "libcef.so alredy installed in \"" $ENV{EXTERNLIBS}/ALL/ "\": this can cause runtime conflicts if a different cef version is being installed.\n
          Clear all cef related links from " $ENV{EXTERNLIBS}/ALL/ " to ensure proper functionality.")
        endif()

        #temporary directory to store download
        file(MAKE_DIRECTORY $ENV{EXTERNLIBS}/cef-stage)
        
        set(CEF_URL https://cef-builds.spotifycdn.com/cef_binary_${CEF_VERSION}_linux64.tar.bz2)
        message("downloading from " ${CEF_URL})
        #get sha1 hash
        set(CEF_CHECKSUM_FILE $ENV{EXTERNLIBS}/cef-stage/hash.sha1)
        file(DOWNLOAD "${CEF_URL}.sha1" ${CEF_CHECKSUM_FILE})
        file(READ "${CEF_CHECKSUM_FILE}" CEF_CHECKSUM)
        #prepare install directory
        if(EXISTS $ENV{EXTERNLIBS}/cef)
          message("clearing old cef dir")  
          file(REMOVE_RECURSE $ENV{EXTERNLIBS}/cef)
        else()
          file(MAKE_DIRECTORY $ENV{EXTERNLIBS}/cef)
        endif()

        #download
        file(DOWNLOAD "${CEF_URL}" EXPECTED_HASH SHA1=${CEF_CHECKSUM} $ENV{EXTERNLIBS}/cef-stage/cef_binary_${CEF_VERSION}_linux64.tar.bz2)
        #extract
        include(FetchContent)
        file(ARCHIVE_EXTRACT INPUT $ENV{EXTERNLIBS}/cef-stage/cef_binary_${CEF_VERSION}_linux64.tar.bz2 DESTINATION $ENV{EXTERNLIBS}/cef-stage)
        file(RENAME $ENV{EXTERNLIBS}/cef-stage/cef_binary_${CEF_VERSION}_linux64 $ENV{EXTERNLIBS}/cef/)  
        #move archive to install dir as reference
        file(RENAME $ENV{EXTERNLIBS}/cef-stage/cef_binary_${CEF_VERSION}_linux64.tar.bz2 $ENV{EXTERNLIBS}/cef/cef_binary_${CEF_VERSION}_linux64.tar.bz2)
        #use stow to creaty runtime symlinks
        EXECUTE_PROCESS(
          COMMAND  ${CMAKE_CURRENT_SOURCE_DIR}/install_cef.sh $ENV{EXTERNLIBS}
        )
        #cleanup stage 
        file(REMOVE_RECURSE $ENV{EXTERNLIBS}/cef-stage)
        file(REMOVE $ENV{EXTERNLIBS}/cef-stage)
      else()
        message("chromium embedded framework version " ${CEF_VERSION} " is already installed")  
      endif()

    endif()
    else()
      message("can not build CEF: EXTERNLIBS environment variable not set")
    endif()
endif()

#find CEF
  IF (NOT CEF_ROOT)
    SET(CEF_ROOT $ENV{EXTERNLIBS}/cef)
    file(TO_CMAKE_PATH "$ENV{EXTERNLIBS}/cef" CEF_ROOT)
  ENDIF()

  # Require the CEF wrapper build directory
  if(NOT IS_DIRECTORY "${CEF_ROOT}/build/libcef_dll_wrapper" AND
     NOT IS_DIRECTORY "${CEF_ROOT}/libcef_dll_wrapper")
    USING_MESSAGE("Skipping because libcef_dll_wrapper not found at ${CEF_ROOT}/build/libcef_dll_wrapper or ${CEF_ROOT}/libcef_dll_wrapper")
    return()
  endif()

 message("looking for CEF in " ${CEF_ROOT}) 
find_package(CEF)
if(NOT CEF_INCLUDE_PATH OR 
   NOT CEF_BINARY_DIR_DEBUG OR 
   NOT CEF_BINARY_DIR_RELEASE)
  message(STATUS "Skipping Browser plugin: CEF not found at ${CEF_ROOT}")
  message(STATUS "Make sure ${CEF_ROOT}/cmake contains cef_variables.cmake and cef_macros.cmake")
  return()
endif()

# Check if wrapper library exists
if(NOT IS_DIRECTORY "${_CEF_ROOT}/libcef_dll_wrapper")
  message(STATUS "Skipping Browser plugin: libcef_dll_wrapper not found")
  message(STATUS "Build CEF wrapper: cd ${CEF_ROOT} && cmake . && cmake --build . --config Debug/Release")
  return()
endif()


message(STATUS "Building Browser plugin with CEF from ${CEF_ROOT}")

if(WIN32)
  #expect debug cef wrapper to be build with debug post fix 'd'
  set(CEF_WRAPPER_LIB_DEBUG   "${_CEF_ROOT}/libcef_dll_wrapper/Debug/libcef_dll_wrapperd.lib")
  set(CEF_WRAPPER_LIB_RELEASE "${_CEF_ROOT}/libcef_dll_wrapper/Release/libcef_dll_wrapper.lib")
  set(CEF_LIBCEF_LIB          "${CEF_LIB_RELEASE}")  # Import lib for libcef.dll
  
  set(CEF_LIBRARIES
    optimized ${CEF_WRAPPER_LIB_RELEASE}
    debug     ${CEF_WRAPPER_LIB_DEBUG}
    ${CEF_LIBCEF_LIB}
    ${CEF_STANDARD_LIBS}
  )
else()
  set(CEF_WRAPPER_LIB_DEBUG   "${_CEF_ROOT}/libcef_dll_wrapper/libcef_dll_wrapper_debug.a")
  set(CEF_WRAPPER_LIB_RELEASE "${_CEF_ROOT}/libcef_dll_wrapper/libcef_dll_wrapper.a")
  
  set(CEF_LIBRARIES
    optimized ${CEF_WRAPPER_LIB_RELEASE}
    debug     ${CEF_WRAPPER_LIB_DEBUG}
    ${CEF_LIB_RELEASE}
  )
  
  # Set rpath for libcef.so discovery
  set(CMAKE_BUILD_RPATH   "${CEF_RESOURCE_DIR}")
  set(CMAKE_INSTALL_RPATH "${CEF_RESOURCE_DIR}")
endif()

set(EXTRA_LIBS ${EXTRA_LIBS} ${CEF_LIBRARIES})
message("EXTRA_LIBS: " ${EXTRA_LIBS})
SET(HEADERS
  CEF.h
  CEFkeyssymdef.h
)
SET(SOURCES
  CEF.cpp
)
message("CEF_INCLUDE_PATH: " ${CEF_INCLUDE_PATH})
message(STATUS "CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
cover_add_plugin(Browser)
target_include_directories(Browser PRIVATE ${CEF_INCLUDE_PATH})

# if(${BUILD_CEF})
#   add_dependencies(Browser cef_build)
# endif()

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
target_include_directories(CEFBrowserHelper PRIVATE ${CEF_INCLUDE_PATH})
if(MSVC)
  set_property(TARGET CEFBrowserHelper PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  set_property(TARGET Browser PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  target_compile_options(CEFBrowserHelper PRIVATE $<$<CONFIG:Debug>:/MDd> $<$<CONFIG:Release>:/MD>)
  target_compile_options(Browser PRIVATE $<$<CONFIG:Debug>:/MDd> $<$<CONFIG:Release>:/MD>)
endif()