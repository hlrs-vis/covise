PROJECT(RidgeSurfacePlugin)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../modules/impl/ridge_surface)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../)

ADD_PARAVIEW_PLUGIN(
  UNIVIZ_RidgeSurface_SMPlugin "1.0"
  SERVER_MANAGER_XML 
    ${CMAKE_CURRENT_SOURCE_DIR}/../RidgeSurface_Server.xml
  SERVER_MANAGER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../vtkRidgeSurface.cxx
  SOURCES 
)

TARGET_LINK_LIBRARIES(
  UNIVIZ_RidgeSurface_SMPlugin
  ${UNIVIZ_PLUGIN_LIBS}
)
  
ADD_PARAVIEW_PLUGIN(
  UNIVIZ_RidgeSurface_GUIPlugin
  "1.0"
  GUI_RESOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/RidgeSurface.qrc 
)

