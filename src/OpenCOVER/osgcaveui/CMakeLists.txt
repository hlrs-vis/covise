# OpenCOVER osgcaveui
USING(Virvo)
USING(VTK optional)
IF(NOT OPENSCENEGRAPH_FOUND)
  RETURN()
ENDIF()

USING(CG optional)

ADD_DEFINITIONS(-DCOMPILE_CUI)

ADD_DEFINITIONS(-DCOVER)

INCLUDE_DIRECTORIES(
  ${OPENGL_INCLUDE_DIR} 
  ${XERCESC_INCLUDE_DIR}
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(OSGCAVEUI_HEADERS
  Interface.h
  Arrow.h
  Dial.h
  LogFile.h
  RadioGroup.h
  Bar.h
  DigitLabel.h
  FileBrowser.h
  Marker.h
  Rectangle.h
  Button.h
  DigitPanel.h
  Measure.h
  TextureWidget.h
  Calculator.h
  Events.h
  Message.h
  Widget.h
  Card.h
  FloatOMeter.h
  Paintbrush.h
  WidgetInfo.h
  CheckBox.h
  HeightFieldPickBox.h
  Panel.h
  ColorWheel.h
  InputDevice.h
  PickBox.h
  CUI.h
  Interaction.h
  RadioButton.h
)

SET(SOURCES
  Interface.cpp
  Arrow.cpp
  CUI.cpp
  Interaction.cpp
  PickBox.cpp
  Bar.cpp
  Dial.cpp
  FileBrowser.cpp
  LogFile.cpp
  RadioButton.cpp
  Button.cpp
  DigitLabel.cpp
  Marker.cpp
  RadioGroup.cpp
  Calculator.cpp
  DigitPanel.cpp
  Measure.cpp
  Rectangle.cpp
  Card.cpp
  FloatOMeter.cpp
  Message.cpp
  TextureWidget.cpp
  CheckBox.cpp
  HeightFieldPickBox.cpp
  Paintbrush.cpp
  Widget.cpp
  ColorWheel.cpp
  InputDevice.cpp
  Panel.cpp
  WidgetInfo.cpp
)

ADD_COVISE_LIBRARY(osgcaveui SHARED )
TARGET_LINK_LIBRARIES(osgcaveui coOpenCOVER coOpenVRUI coOSGVRUI
   ${XERCESC_LIBRARIES} ${OPENSCENEGRAPH_LIBRARIES} ${OPENGL_LIBRARIES} ${TIFF_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${EXTRA_LIBS})
COVISE_INSTALL_TARGET(osgcaveui)
COVISE_INSTALL_HEADERS(osgcaveui ${OSGCAVEUI_HEADERS})
