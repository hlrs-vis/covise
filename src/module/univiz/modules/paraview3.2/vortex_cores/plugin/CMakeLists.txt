PROJECT(VorextCoresPlugin)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../modules/impl/vortex_cores)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../)

ADD_PARAVIEW_PLUGIN(
  UNIVIZ_VortexCores_SMPlugin "1.0"
  SERVER_MANAGER_XML 
    ${CMAKE_CURRENT_SOURCE_DIR}/../VortexCores_Server.xml
  SERVER_MANAGER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../vtkVortexCores.cxx
  SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../modules/impl/vortex_cores/computeVortexCores.cpp
)

TARGET_LINK_LIBRARIES(
  UNIVIZ_VortexCores_SMPlugin
  ${UNIVIZ_PLUGIN_LIBS}
)
  
ADD_PARAVIEW_PLUGIN(
  UNIVIZ_VortexCores_GUIPlugin
  "1.0"
  GUI_RESOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/VortexCores.qrc 
)

