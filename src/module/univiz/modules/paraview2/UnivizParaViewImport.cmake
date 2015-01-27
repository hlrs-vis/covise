######################################################################
# in the example replace VortexCores with the name of your module
# Univiz_SOURCE_DIR will already be set by ParaView
######################################################################

ADD_DEFINITIONS(-DCSCS_PARAVIEW_INTERNAL)
ADD_DEFINITIONS(-DVTK)

# Additional libraries (Univiz)

# Define the source files that should be built
SET (Univiz_SRCS  
  ${Univiz_SOURCE_DIR}/../../libs/unstructured/unstructured.cpp
  ${Univiz_SOURCE_DIR}/../../libs/unifield/unifield.cpp
  ${Univiz_SOURCE_DIR}/../../libs/unigeom/unigeom.cpp
  ${Univiz_SOURCE_DIR}/../../libs/unisys/unisys.cpp
  ${Univiz_SOURCE_DIR}/../../libs/paraview_ext/paraview_ext.cpp 
  ${Univiz_SOURCE_DIR}/../impl/vortex_cores/computeVortexCores.cpp
)

# Define sources that should be built and wrapped in the client server
SET (Univiz_WRAPPED_SRCS 
  ${Univiz_SOURCE_DIR}/field_to_lines/vtkFieldToLines.cxx
  ${Univiz_SOURCE_DIR}/FLE/vtkFLE.cxx
  ${Univiz_SOURCE_DIR}/flow_topo/vtkFlowTopo.cxx
  ${Univiz_SOURCE_DIR}/ridge_surface/vtkRidgeSurface.cxx
  ${Univiz_SOURCE_DIR}/vortex_cores/vtkVortexCores.cxx
  ${Univiz_SOURCE_DIR}/vortex_criteria/vtkVortexCriteria.cxx
)

INCLUDE_DIRECTORIES(
  ${Univiz_SOURCE_DIR}
  ${Univiz_SOURCE_DIR}/../../libs/linalg
  ${Univiz_SOURCE_DIR}/../../libs/unifield
  ${Univiz_SOURCE_DIR}/../../libs/unstructured
  ${Univiz_SOURCE_DIR}/../../libs/unigeom
  ${Univiz_SOURCE_DIR}/../../libs/unisys
  ${Univiz_SOURCE_DIR}/../../libs/paraview_ext
  ${Univiz_SOURCE_DIR}/../impl/field_to_lines
  ${Univiz_SOURCE_DIR}/field_to_lines
  ${Univiz_SOURCE_DIR}/../impl/FLE
  ${Univiz_SOURCE_DIR}/FLE
  ${Univiz_SOURCE_DIR}/../impl/flow_topo
  ${Univiz_SOURCE_DIR}/flow_topo
  ${Univiz_SOURCE_DIR}/../impl/ridge_surface
  ${Univiz_SOURCE_DIR}/ridge_surface
  ${Univiz_SOURCE_DIR}/../impl/vortex_cores
  ${Univiz_SOURCE_DIR}/vortex_cores
  ${Univiz_SOURCE_DIR}/../impl/vortex_criteria
  ${Univiz_SOURCE_DIR}/vortex_criteria
)

# invoke this macro to add the sources to paraview and wrap them for the
# client server 
PARAVIEW_INCLUDE_WRAPPED_SOURCES("${Univiz_WRAPPED_SRCS}")

# invoke this macro to add link libraries to PV
#PARAVIEW_LINK_LIBRARIES("${CFX_LIBS}")

# invoke this macro to add the sources to the build (but not wrap them into
# the client server
PARAVIEW_INCLUDE_SOURCES("${Univiz_SRCS}")

# invoke this macro to add sources into the client and also wrap them into Tcl
#PARAVIEW_INCLUDE_CLIENT_SOURCES("${Univiz_GUI_SRCS}")

PARAVIEW_INCLUDE_GUI_RESOURCES("${Univiz_SOURCE_DIR}/Univiz_Client.xml")
IF(PARAVIEW_PRE_2_6)
PARAVIEW_INCLUDE_SERVERMANAGER_RESOURCES("${Univiz_SOURCE_DIR}/Univiz_Server.xml.paraview_2.4")
ELSE(PARAVIEW_PRE_2_6)
PARAVIEW_INCLUDE_SERVERMANAGER_RESOURCES("${Univiz_SOURCE_DIR}/Univiz_Server.xml")
ENDIF(PARAVIEW_PRE_2_6)
