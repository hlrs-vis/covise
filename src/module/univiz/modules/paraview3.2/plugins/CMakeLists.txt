PROJECT("UniViz Plugins")

#--------------------------------------------------
# Find Paraview once for all modules
#--------------------------------------------------
FIND_PACKAGE(ParaView REQUIRED)

INCLUDE(${PARAVIEW_USE_FILE})

#--------------------------------------------------
# Macro to add modules
#--------------------------------------------------
MACRO(UNIVIZ_PLUGIN name path)
  OPTION(UNIVIZ_PLUGIN_${name} "Build ${name} plugin" helpstring ON)
  IF(UNIVIZ_PLUGIN_${name})
    SET(UNIVIZ_PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../${path}")
    MESSAGE("Setting Plugin Directory to ${UNIVIZ_PLUGIN_DIR}")
    GET_FILENAME_COMPONENT(TEMP_Filename ${path} NAME)
    ADD_SUBDIRECTORY("${UNIVIZ_PLUGIN_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
  ENDIF(UNIVIZ_PLUGIN_${name})
ENDMACRO(UNIVIZ_PLUGIN)

#--------------------------------------------------
# Flag for modules to know vtk target
#--------------------------------------------------
ADD_DEFINITIONS(-DVTK)
ADD_DEFINITIONS(-DCSCS_PARAVIEW_INTERNAL)

#--------------------------------------------------
# Build common libraries shared between plugins
#--------------------------------------------------
ADD_LIBRARY(
  Unstructured SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/unstructured/unstructured.cpp
)

ADD_LIBRARY(
  Unigeom SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/unigeom/unigeom.cpp
)

ADD_LIBRARY(
  Unisys SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/unisys/unisys.cpp
)

ADD_LIBRARY(
  Unifield SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/unifield/unifield.cpp
)

ADD_LIBRARY(
  ParaviewExt SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/paraview_ext/paraview_ext.cpp
)

ADD_LIBRARY(
  Util SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/util/util.cpp
)


SET(UNIVIZ_PLUGIN_LIBS 
  Unstructured
  Unigeom 
  Unisys 
  Unifield 
  ParaviewExt
  Util
)

#--------------------------------------------------
# Include directories
#--------------------------------------------------

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/linalg)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/unifield)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/unigeom)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/unstructured)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/unisys)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/paraview_ext)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/util)

#--------------------------------------------------
# VortexCores
#--------------------------------------------------
UNIVIZ_PLUGIN(VortexCores "vortex_cores/plugin")

#--------------------------------------------------
# VortexCriteria
#--------------------------------------------------
UNIVIZ_PLUGIN(VortexCriteria "vortex_criteria/plugin")

#--------------------------------------------------
# Statistics
#--------------------------------------------------
UNIVIZ_PLUGIN(Statistics "statistics/plugin")

#--------------------------------------------------
# FieldToLines
#--------------------------------------------------
UNIVIZ_PLUGIN(FieldToLines "field_to_lines/plugin")

#--------------------------------------------------
# FLE
#--------------------------------------------------
UNIVIZ_PLUGIN(FLE "FLE/plugin")

#--------------------------------------------------
# FlowTopo
#--------------------------------------------------
UNIVIZ_PLUGIN(FlowTopo "flow_topo/plugin")

#--------------------------------------------------
# RidgeSurface
#--------------------------------------------------
# TODO: support Windows
IF (NOT WIN32)
UNIVIZ_PLUGIN(RidgeSurface "ridge_surface/plugin")
ENDIF (NOT WIN32)

#--------------------------------------------------
# WriteUnstructured
#--------------------------------------------------
UNIVIZ_PLUGIN(WriteUnstructured "write_unstructured/plugin")

#--------------------------------------------------
# ReadUnstructured
#--------------------------------------------------
UNIVIZ_PLUGIN(ReadUnstructured "read_unstructured/plugin")

