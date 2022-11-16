# @file
# 
# CMakeLists.txt for ManagedOpenCOVER
#
# @author Blasius Czink

USING(Surface)
if(${CMAKE_GENERATOR} STREQUAL "Ninja")
    message("not building ManagedOpenCOVER with Ninja")
    RETURN()
endif()
IF(NOT WIN32)
  RETURN()
ENDIF(NOT WIN32)


  INCLUDE_DIRECTORIES(SYSTEM
    ${ZLIB_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIR}
    ${PNG_INCLUDE_DIR}
    ${TIFF_INCLUDE_DIR}
    ${OPENSCENEGRAPH_INCLUDE_DIRS}
  )
  INCLUDE_DIRECTORIES(
    "${COVISEDIR}/src/OpenCOVER"
  )
SET(SOURCES
  ManagedOpenCOVER.cpp
  AssemblyInfo.cpp
)

SET(HEADERS
  ManagedOpenCOVER.h
)

#set_source_files_properties(ManagedOpenCOVER.cpp PROPERTIES COMPILE_FLAGS "/CLR /TP")

ADD_COVISE_LIBRARY(ManagedOpenCOVER SHARED ${EXTRASOURCES} ${SOURCES} ${HEADERS})
set(EXTRA_LIBS ${EXTRA_LIBS} ws2_32.lib)

#set_target_properties(ManagedOpenCOVER PROPERTIES VS_DOTNET_REFERENCES "${CMAKE_CURRENT_SOURCE_DIR}/../OrionMaster/3rdParty/GMap.NET.Core.dll;System;System.Core;System.Data;System.Drawing;System.Xml;WindowsBase")
set_target_properties(ManagedOpenCOVER PROPERTIES COMPILE_FLAGS "/clr /EHa")
#set_target_properties(ManagedOpenCOVER PROPERTIES DEBUG_POSTFIX "d")

if(CMAKE_CXX_FLAGS_DEBUG MATCHES "/RTC1")
   string(REPLACE "/RTC1" " " CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

if(CMAKE_CXX_FLAGS MATCHES "/EHsc")
   string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
set_property(TARGET ManagedOpenCOVER PROPERTY COMMON_LANGUAGE_RUNTIME "")
set_property(TARGET ManagedOpenCOVER PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "v4.7.1")

set_property(TARGET ManagedOpenCOVER PROPERTY VS_DOTNET_REFERENCES
    "PresentationCore"
	"C:/src/externlibs/zebu/surface/Microsoft.Surface.Core.dll"
    "System"
    "System.Core"
    "System.Data"
    "System.Drawing"
    "System.Windows.Forms"
    "System.Windows.Presentation"
    "System.Xaml"
    "System.Xml"
    "WindowsBase")
TARGET_LINK_LIBRARIES(ManagedOpenCOVER coOpenPluginUtil coOpenCOVER coOpenVRUI coOSGVRUI
     ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_CONFIG_LIBRARY} ${COVISE_UTIL_LIBRARY}
     ${OPENSCENEGRAPH_LIBRARIES} ${EXTRA_LIBS}) 
	 
COVISE_INSTALL_TARGET(ManagedOpenCOVER)
