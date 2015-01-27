PROJECT(WriteUnstructuredPlugin)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../modules/impl/write_unstructured)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../)

ADD_PARAVIEW_PLUGIN(
  UNIVIZ_WriteUnstructured_SMPlugin "1.0"
  SERVER_MANAGER_XML 
    ${CMAKE_CURRENT_SOURCE_DIR}/../WriteUnstructured_Server.xml
  SERVER_MANAGER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../vtkWriteUnstructured.cxx
  SOURCES 
)

TARGET_LINK_LIBRARIES(
  UNIVIZ_WriteUnstructured_SMPlugin
  ${UNIVIZ_PLUGIN_LIBS}
)
  
ADD_PARAVIEW_PLUGIN(
  UNIVIZ_WriteUnstructured_GUIPlugin
  "1.0"
  GUI_RESOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/WriteUnstructured.qrc 
)

