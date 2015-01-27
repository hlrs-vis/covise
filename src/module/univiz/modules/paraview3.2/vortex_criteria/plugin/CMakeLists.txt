PROJECT(VorextCriteriaPlugin)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../modules/impl/vortex_criteria)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../)

ADD_PARAVIEW_PLUGIN(
  UNIVIZ_VortexCriteria_SMPlugin "1.0"
  SERVER_MANAGER_XML 
    ${CMAKE_CURRENT_SOURCE_DIR}/../VortexCriteria_Server.xml
  SERVER_MANAGER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../vtkVortexCriteria.cxx
  SOURCES 
)

TARGET_LINK_LIBRARIES(
  UNIVIZ_VortexCriteria_SMPlugin
  ${UNIVIZ_PLUGIN_LIBS}
)
  
ADD_PARAVIEW_PLUGIN(
  UNIVIZ_VortexCriteria_GUIPlugin
  "1.0"
  GUI_RESOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/VortexCriteria.qrc 
)

