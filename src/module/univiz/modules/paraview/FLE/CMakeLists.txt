# Set the project/library name here.  Classes should be declared as
# "class VTK_PVLocal_EXPORT vtkFoo", where PVLocal is the name of the
# project set here.
PROJECT(FLE)

ADD_DEFINITIONS(-DVTK)

# Additional libraries (Univiz)
SET(BUILD_SHARED_LIBS ON)
ADD_LIBRARY(Unstructured ../../../libs/unstructured/unstructured.cpp)
ADD_LIBRARY(Unifield ../../../libs/unifield/unifield.cpp)
ADD_LIBRARY(Unisys ../../../libs/unisys/unisys.cpp)
ADD_LIBRARY(Paraview_Ext ../../../libs/paraview_ext/paraview_ext.cpp)
ADD_LIBRARY(Util ../../../libs/util/util.cpp)
SET(ADDITIONAL_LIBS Unstructured Unifield Unisys Paraview_Ext Util)

# Set your list of sources here.  Do not change the name of the
# PVLocal_SRCS variable.
SET(PVLocal_SRCS
  vtkFLE.cxx
)

# List the kits from VTK that are needed by this project.  Do not
# change the name of the PVLocal_LIBS variable.
SET(PVLocal_LIBS
  vtkGraphics
)

# Univiz
INCLUDE_DIRECTORIES(../../../libs/linalg)
INCLUDE_DIRECTORIES(../../../libs/unstructured)
INCLUDE_DIRECTORIES(../../../libs/unifield)
INCLUDE_DIRECTORIES(../../../libs/unisys)
INCLUDE_DIRECTORIES(../../../libs/paraview_ext)
INCLUDE_DIRECTORIES(../../../libs/util)
INCLUDE_DIRECTORIES(../../impl/FLE)

#-----------------------------------------------------------------------------
# Most users should not need to change anything below this line.

# Need to include class headers and the configuration header.
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR})

# Import ParaView build settings.
FIND_PACKAGE(ParaView)
IF(NOT ParaView_FOUND)
  MESSAGE(FATAL_ERROR "ParaView is required.  Set ParaView_DIR.")
ENDIF(NOT ParaView_FOUND)
IF(NOT PARAVIEW_BUILD_SHARED_LIBS)
  MESSAGE(FATAL_ERROR "ParaView must be built with BUILD_SHARED_LIBS ON.")
ENDIF(NOT PARAVIEW_BUILD_SHARED_LIBS)
INCLUDE(${PARAVIEW_USE_FILE})

# Configure output directories.
SET (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR} CACHE INTERNAL
     "Single output directory for building all libraries.")
SET (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR} CACHE INTERNAL
     "Single output directory for building all executables.")

# Create the configuration header.
CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/vtkPVLocalConfigure.h.in
               ${PROJECT_BINARY_DIR}/vtk${PROJECT_NAME}Configure.h
               @ONLY IMMEDIATE)

# Create vtk client/server wrappers for the classes.
VTK_WRAP_ClientServer(${PROJECT_NAME} PVLocalCS_SRCS "${PVLocal_SRCS}")

# Build the package as a plugin for ParaView.
ADD_LIBRARY(${PROJECT_NAME} MODULE ${PVLocal_SRCS} ${PVLocalCS_SRCS})
#TARGET_LINK_LIBRARIES(${PROJECT_NAME} vtkClientServer)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} vtkClientServer ${ADDITIONAL_LIBS})
FOREACH(c ${PVLocal_LIBS})
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${c}CS)
ENDFOREACH(c)

# Place the package configuration file into the build tree.
CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/PVLocal.xml.in
               ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.xml @ONLY IMMEDIATE)

# Place the package configuration file into the build tree.
IF(PARAVIEW_PRE_2_6)
CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/PVLocal.pvsm.in.paraview_2.4
               ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.pvsm @ONLY IMMEDIATE)
ELSE(PARAVIEW_PRE_2_6)
CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/PVLocal.pvsm.in
               ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.pvsm @ONLY IMMEDIATE)
ENDIF(PARAVIEW_PRE_2_6)
