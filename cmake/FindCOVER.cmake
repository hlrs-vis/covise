# - Find COVER

find_package(COVISE)
if(NOT COVISE_FOUND)
   message("COVER: COVISE not found")
   return()
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "$ENV{EXTERNLIBS}/OpenSceneGraph")
covise_find_package(OpenSceneGraph 3.2.0 COMPONENTS osgViewer osgGA osgDB)
find_package(OpenGL)
if(NOT OPENSCENEGRAPH_FOUND)
   message("COVER: OpenSceneGraph not found")
   return()
endif()

if(COVER_INCLUDE_DIR)
   set(COVER_FIND_QUIETLY TRUE)
endif()

set(COVER_DIR "${COVISEDIR}/src/OpenCOVER")

find_path(COVER_INCLUDE_DIR "cover/coVRPluginSupport.h"
   PATHS
   ${COVER_DIR}
   DOC "COVER - Headers"
)

find_path(COVISE_EXPORTS_FILEPATH "cover-exports.cmake"
   PATHS
   ${COVISEDIR}/${COVISE_ARCHSUFFIX}
   DOC "COVER - CMake library exports"
)
if (COVISE_EXPORTS_FILEPATH)
   set(COVISE_EXPORTS_IMPORT "${COVISE_EXPORTS_FILEPATH}/covise-exports.cmake")
   include(${COVISE_EXPORTS_IMPORT})
   set(COVISE_OPTIONS_IMPORT "${COVISE_EXPORTS_FILEPATH}/CoviseOptions.cmake")
   include(${COVISE_OPTIONS_IMPORT})
else()
   message("COVER: CMake library exports file not found")
endif()

covise_find_library(COVER coOpenCOVER)
covise_find_library(COVER_VRUI coOpenVRUI)
covise_find_library(COVER_OSGVRUI coOSGVRUI)
covise_find_library(COVER_PLUGINUTIL coOpenPluginUtil)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(COVER DEFAULT_MSG
   COVER_DIR
   COVER_LIBRARY
   COVER_VRUI_LIBRARY
   COVER_OSGVRUI_LIBRARY
   COVER_PLUGINUTIL_LIBRARY
   COVISE_VRBCLIENT_LIBRARY
   COVISE_CONFIG_LIBRARY
   COVISE_UTIL_LIBRARY
   COVER_INCLUDE_DIR COVISE_INCLUDE_DIR)
mark_as_advanced(COVER_LIBRARY
   COVER_VRUI_LIBRARY
   COVER_OSGVRUI_LIBRARY
   COVER_PLUGINUTIL_LIBRARY
   COVER_INCLUDE_DIR)

if(COVER_FOUND)
   set(COVER_INCLUDE_DIRS ${COVER_INCLUDE_DIR} ${COVISE_INCLUDE_DIR})
endif()

MACRO(COVER_ADD_PLUGIN targetname)
   COVER_ADD_PLUGIN_TARGET(${targetname} ${ARGN})
  IF(DEFINED EXTRA_LIBS)
    TARGET_LINK_LIBRARIES(${targetname} ${EXTRA_LIBS})
  ENDIF()
  if(CUDA_FOUND AND COVISE_USE_CUDA)
     TARGET_LINK_LIBRARIES(${targetname} ${CUDA_LIBRARIES})
  endif()
  COVER_INSTALL_PLUGIN(${targetname})
ENDMACRO(COVER_ADD_PLUGIN)

# Macro to add OpenCOVER plugins
MACRO(COVER_ADD_PLUGIN_TARGET targetname)
  #USING(VTK optional)
  
  IF(WIN32)
    ADD_DEFINITIONS(-DIMPORT_PLUGIN)
  ENDIF(WIN32)

  INCLUDE_DIRECTORIES(
    ${ZLIB_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIR}
    ${PNG_INCLUDE_DIR}
    ${TIFF_INCLUDE_DIR}
    ${OPENSCENEGRAPH_INCLUDE_DIRS}
    ${COVER_INCLUDE_DIRS}
  )

  set(LIBRARY_OUTPUT_PATH "${LIBRARY_OUTPUT_PATH}/OpenCOVER/plugins")
  if(UNIX AND NOT APPLE)
     ADD_LIBRARY(${targetname} SHARED ${ARGN})
   else()
      ADD_LIBRARY(${targetname} MODULE ${ARGN})
   endif()
  
  set_target_properties(${targetname} PROPERTIES FOLDER "OpenCOVER_Plugins")
  
  # SET_TARGET_PROPERTIES(${targetname} PROPERTIES PROJECT_LABEL "${targetname}")
  SET_TARGET_PROPERTIES(${targetname} PROPERTIES OUTPUT_NAME "${targetname}")

  COVISE_ADJUST_OUTPUT_DIR(${targetname} "OpenCOVER/plugins")
  
  # set additional COVISE_COMPILE_FLAGS
  SET_TARGET_PROPERTIES(${targetname} PROPERTIES COMPILE_FLAGS "${COVISE_COMPILE_FLAGS}")
  # set additional COVISE_LINK_FLAGS
  #SET_TARGET_PROPERTIES(${targetname} PROPERTIES LINK_FLAGS "${COVISE_LINK_FLAGS}")
  # switch off "lib" prefix for MinGW
  IF(MINGW)
    SET_TARGET_PROPERTIES(${targetname} PROPERTIES PREFIX "")
  ENDIF(MINGW)

  IF(APPLE)
     COVISE_ADD_LINK_FLAGS(${targetname} "-undefined error")
     COVISE_ADD_LINK_FLAGS(${targetname} "-flat_namespace")
  ENDIF(APPLE)
    
  TARGET_LINK_LIBRARIES(${targetname} ${COVER_PLUGINUTIL_LIBRARY}
     ${COVER_LIBRARY} ${COVER_VRUI_LIBRARY} ${COVER_OSGVRUI_LIBRARY}
     ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_CONFIG_LIBRARY} ${COVISE_UTIL_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES})
  
  qt_use_modules(${targetname} Core)
ENDMACRO(COVER_ADD_PLUGIN_TARGET)

# Macro to install an OpenCOVER plugin
MACRO(COVER_INSTALL_PLUGIN targetname)
  INSTALL(TARGETS ${ARGV} EXPORT covise-targets
     RUNTIME DESTINATION covise/${COVISE_ARCHSUFFIX}/lib/OpenCOVER/plugins
     LIBRARY DESTINATION covise/${COVISE_ARCHSUFFIX}/lib/OpenCOVER/plugins
     ARCHIVE DESTINATION covise/${COVISE_ARCHSUFFIX}/lib/OpenCOVER/plugins
          COMPONENT osgplugins.${category}
  )
  COVISE_INSTALL_DEPENDENCIES(${targetname})
ENDMACRO(COVER_INSTALL_PLUGIN)
