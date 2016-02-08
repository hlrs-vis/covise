# @file
# 
# CMakeLists.txt for covise - kernel - OpenVRUI - osg (coOSGVRUI)
#
# @author Blasius Czink

IF(NOT OPENSCENEGRAPH_FOUND)
  RETURN()
ENDIF()

ADD_DEFINITIONS(-DCOVISE_OSGVRUI)

INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
  ../
  ../osg
  ../sginterface
  ../util
)

SET(OSGVRUI_SOURCES
  NodeDumpVisitor.cpp
  OSGVruiColoredBackground.cpp
  OSGVruiDefaultButtonGeometry.cpp
  OSGVruiFlatButtonGeometry.cpp
  OSGVruiFlatPanelGeometry.cpp
  OSGVruiFrame.cpp
  OSGVruiHit.cpp
  OSGVruiLabel.cpp
  OSGVruiMatrix.cpp
  OSGVruiNode.cpp
  OSGVruiNull.cpp
  OSGVruiPanelGeometry.cpp
  OSGVruiPresets.cpp
  OSGVruiRectButtonGeometry.cpp
  OSGVruiSlider.cpp
  OSGVruiSquareButtonGeometry.cpp
  OSGVruiTexturedBackground.cpp
  OSGVruiTextureRectBackground.cpp
  OSGVruiToggleButtonGeometry.cpp
  OSGVruiTransformNode.cpp
  OSGVruiUserDataCollection.cpp
  OSGVruiUIContainer.cpp
  OSGVruiUIElement.cpp
  OSGVruiValuePoti.cpp
  OSGVruiTextButtonGeometry.cpp
  mathUtils.cpp
)

SET(OSGVRUI_HEADERS
  NodeDumpVisitor.h
  OSGVruiColoredBackground.h
  OSGVruiDefaultButtonGeometry.h
  OSGVruiFlatButtonGeometry.h
  OSGVruiFlatPanelGeometry.h
  OSGVruiFrame.h
  OSGVruiHit.h
  OSGVruiLabel.h
  OSGVruiMatrix.h
  OSGVruiNode.h
  OSGVruiNull.h
  OSGVruiPanelGeometry.h
  OSGVruiPresets.h
  OSGVruiRectButtonGeometry.h
  OSGVruiSlider.h
  OSGVruiSquareButtonGeometry.h
  OSGVruiTexturedBackground.h
  OSGVruiTextureRectBackground.h
  OSGVruiToggleButtonGeometry.h
  OSGVruiTransformNode.h
  OSGVruiUserDataCollection.h
  OSGVruiUIContainer.h
  OSGVruiUIElement.h
  OSGVruiValuePoti.h
  OSGVruiTextButtonGeometry.h
  mathUtils.h
)

ADD_COVISE_LIBRARY(coOSGVRUI SHARED ${OSGVRUI_SOURCES} ${OSGVRUI_HEADERS})
TARGET_LINK_LIBRARIES(coOSGVRUI coOpenVRUI ${COVISE_CONFIG_LIBRARY} ${COVISE_UTIL_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES})
COVISE_INSTALL_TARGET(coOSGVRUI)
COVISE_INSTALL_HEADERS(OpenVRUI/osg ${OSGVRUI_HEADERS})
