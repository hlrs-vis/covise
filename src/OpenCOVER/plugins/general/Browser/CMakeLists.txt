#OpenCOVER plugins general Browser based on Chromium Embedded Framework
if(NOT WIN32 AND NOT APPLE)
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

      #temporary directory to store downloa
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
endif()

USING(CEF)

message("building cef!!!!!!!!!!")
SET(HEADERS
  CEF.h
  CEFkeyssymdef.h
)
SET(SOURCES
  CEF.cpp
)
cover_add_plugin(Browser)
set_property(TARGET Browser PROPERTY CXX_STANDARD 14)

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
set_property(TARGET CEFBrowserHelper PROPERTY CXX_STANDARD 14)
TARGET_LINK_LIBRARIES(CEFBrowserHelper ${EXTRA_LIBS} )

