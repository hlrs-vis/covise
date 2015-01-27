PROJECT(LocalizedFlowPlugin)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../modules/impl/localized_flow)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../)

ADD_PARAVIEW_PLUGIN(
  UNIVIZ_LocalizedFlow_SMPlugin "1.0"
  SERVER_MANAGER_XML 
    ${CMAKE_CURRENT_SOURCE_DIR}/../LocalizedFlow_Server.xml
  SERVER_MANAGER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../vtkLocalizedFlow.cxx
  SOURCES 
)

TARGET_LINK_LIBRARIES(
  UNIVIZ_LocalizedFlow_SMPlugin
  ${UNIVIZ_PLUGIN_LIBS}
)
  
ADD_PARAVIEW_PLUGIN(
  UNIVIZ_LocalizedFlow_GUIPlugin
  "1.0"
  GUI_RESOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/LocalizedFlow.qrc 
)

