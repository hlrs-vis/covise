# @file
# 
# CMakeLists.txt for covise - kernel - OpenVRUI - vsg (coVSGVRUI)
#
#
USING(vsg)

ADD_DEFINITIONS(-DCOVISE_VSGVRUI)

INCLUDE_DIRECTORIES(
  ../
  ../vsg
  ../sginterface
  ../util
)

SET(VSGVRUI_SOURCES
  NodeDumpVisitor.cpp
  VSGVruiColoredBackground.cpp
  VSGVruiDefaultButtonGeometry.cpp
  VSGVruiFlatButtonGeometry.cpp
  VSGVruiFlatPanelGeometry.cpp
  VSGVruiFrame.cpp
  VSGVruiHit.cpp
  VSGVruiLabel.cpp
  VSGVruiMatrix.cpp
  VSGVruiNode.cpp
  VSGVruiNull.cpp
  VSGVruiPanelGeometry.cpp
  VSGVruiPresets.cpp
  VSGVruiRectButtonGeometry.cpp
  VSGVruiSlider.cpp
  VSGVruiSquareButtonGeometry.cpp
  VSGVruiTexturedBackground.cpp
  VSGVruiTextureRectBackground.cpp
  VSGVruiToggleButtonGeometry.cpp
  VSGVruiTransformNode.cpp
  VSGVruiUserDataCollection.cpp
  VSGVruiUIContainer.cpp
  VSGVruiUIElement.cpp
  VSGVruiValuePoti.cpp
  VSGVruiTextButtonGeometry.cpp
  mathUtils.cpp
)

SET(VSGVRUI_HEADERS
  NodeDumpVisitor.h
  VSGVruiColoredBackground.h
  VSGVruiDefaultButtonGeometry.h
  VSGVruiFlatButtonGeometry.h
  VSGVruiFlatPanelGeometry.h
  VSGVruiFrame.h
  VSGVruiHit.h
  VSGVruiLabel.h
  VSGVruiMatrix.h
  VSGVruiNode.h
  VSGVruiNull.h
  VSGVruiPanelGeometry.h
  VSGVruiPresets.h
  VSGVruiRectButtonGeometry.h
  VSGVruiSlider.h
  VSGVruiSquareButtonGeometry.h
  VSGVruiTexturedBackground.h
  VSGVruiTextureRectBackground.h
  VSGVruiToggleButtonGeometry.h
  VSGVruiTransformNode.h
  VSGVruiUserDataCollection.h
  VSGVruiUIContainer.h
  VSGVruiUIElement.h
  VSGVruiValuePoti.h
  VSGVruiTextButtonGeometry.h
  mathUtils.h
)

ADD_COVISE_LIBRARY(coVSGVRUI SHARED ${VSGVRUI_SOURCES} ${VSGVRUI_HEADERS})
TARGET_LINK_LIBRARIES(coVSGVRUI coOpenVRUI ${COVISE_CONFIG_LIBRARY} ${COVISE_UTIL_LIBRARY} ${EXTRA_LIBS})
COVISE_INSTALL_TARGET(coVSGVRUI)
COVISE_INSTALL_HEADERS(OpenVRUI/vsg ${VSGVRUI_HEADERS})
