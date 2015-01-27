# @file
# 
# CMakeLists.txt for covise - kernel - OpenVRUI - base (coOpenVRUI)
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCOVISE_OPENVRUI)

INCLUDE_DIRECTORIES(
  ../
  ../base
  ../sginterface
  ../util
  ../osg
)

SET(BASE_SOURCES
  vruiCollabInterface.cpp
  vruiIntersection.cpp
  vruiRendererInterface.cpp
  vruiVec.cpp
  coAction.cpp
  coBackground.cpp
  coButton.cpp
  coButtonGeometry.cpp
  coButtonMenuItem.cpp
  coCheckboxGroup.cpp
  coCheckboxMenuItem.cpp
  coColoredBackground.cpp
  coDefaultButtonGeometry.cpp
  coDialog.cpp
  coDragNDrop.cpp
  coFlatButtonGeometry.cpp
  coFlatPanelGeometry.cpp
  coFrame.cpp
  coGenericSubMenuItem.cpp
  coIconButtonToolboxItem.cpp
  coIconSubMenuToolboxItem.cpp
  coIconToggleButtonToolboxItem.cpp
  coInteraction.cpp
  coInteractionManager.cpp
  coJoystickManager.cpp
  coLabel.cpp
  coLabelMenuItem.cpp
  coLabelSubMenuToolboxItem.cpp
  coMenu.cpp
  coMenuChangeListener.cpp
  coMenuContainer.cpp
  coMenuItem.cpp
  coNavInteraction.cpp
  coPanel.cpp
  coPanelGeometry.cpp
  coPopupHandle.cpp
  coPotiMenuItem.cpp
  coPotiToolboxItem.cpp
  coProgressBar.cpp
  coProgressBarMenuItem.cpp
  coPushButton.cpp
  coRectButtonGeometry.cpp
  coRotButton.cpp
  coRotPushButton.cpp
  coRotToggleButton.cpp
  coRowContainer.cpp
  coRowMenu.cpp
  coRowMenuHandle.cpp
  coRowMenuItem.cpp
  coSlider.cpp
  coSliderMenuItem.cpp
  coSliderToolboxItem.cpp
  coSlopePoti.cpp
  coSquareButtonGeometry.cpp
  coSubMenuItem.cpp
  coSubMenuToolboxItem.cpp
  coTextButtonGeometry.cpp
  coTexturedBackground.cpp
  coTextureRectBackground.cpp
  coMovableBackgroundMenuItem.cpp
  coToggleButton.cpp
  coToggleButtonGeometry.cpp
  coToolboxMenu.cpp
  coToolboxMenuHandle.cpp
  coToolboxMenuItem.cpp
  coButtonInteraction.cpp
  coTrackerButtonInteraction.cpp
  coMouseButtonInteraction.cpp
  coCombinedButtonInteraction.cpp
  coUpdateManager.cpp
  coUIContainer.cpp
  coUIElement.cpp
  coUIUserData.cpp
  coValuePoti.cpp
#  coDropDownHandle.cpp
#  coImageButtonGeometry.cpp
#  coLabelToggleButtonGeometry.cpp
#  coLabelToggleButtonToolboxItem.cpp
#  coLabelButtonToolboxItem.cpp
)

SET(BASE_HEADERS
  ../sginterface/vruiCollabInterface.h
  ../sginterface/vruiIntersection.h
  ../sginterface/vruiRendererInterface.h
  ../sginterface/vruiVec.h
  ../coAction.h
  ../coBackground.h
  ../coButton.h
  ../coButtonGeometry.h
  ../coButtonMenuItem.h
  ../coCheckboxGroup.h
  ../coCheckboxMenuItem.h
  ../coColoredBackground.h
  ../coDefaultButtonGeometry.h
  ../coDialog.h
  ../coDragNDrop.h
  ../coFlatButtonGeometry.h
  ../coFlatPanelGeometry.h
  ../coFrame.h
  ../coGenericSubMenuItem.h
  ../coIconButtonToolboxItem.h
  ../coIconSubMenuToolboxItem.h
  ../coIconToggleButtonToolboxItem.h
  ../coInteraction.h
  ../coInteractionManager.h
  ../coJoystickManager.h
  ../coLabel.h
  ../coLabelMenuItem.h
  ../coLabelSubMenuToolboxItem.h
  ../coMenu.h
  ../coMenuChangeListener.h
  ../coMenuContainer.h
  ../coMenuItem.h
  ../coNavInteraction.h
  ../coPanel.h
  ../coPanelGeometry.h
  ../coPopupHandle.h
  ../coPotiMenuItem.h
  ../coPotiToolboxItem.h
  ../coProgressBar.h
  ../coProgressBarMenuItem.h
  ../coRectButtonGeometry.h
  ../coRotButton.h
  ../coRowContainer.h
  ../coRowMenu.h
  ../coRowMenuHandle.h
  ../coRowMenuItem.h
  ../coSlider.h
  ../coSliderMenuItem.h
  ../coSliderToolboxItem.h
  ../coSquareButtonGeometry.h
  ../coSubMenuItem.h
  ../coSubMenuToolboxItem.h
  ../coTextButtonGeometry.h
  ../coTexturedBackground.h
  ../coTextureRectBackground.h
  ../coMovableBackgroundMenuItem.h
  ../coToggleButtonGeometry.h
  ../coToolboxMenu.h
  ../coToolboxMenuHandle.h
  ../coToolboxMenuItem.h
  ../coButtonInteraction.h
  ../coTrackerButtonInteraction.h
  ../coMouseButtonInteraction.h
  ../coCombinedButtonInteraction.h
  ../coUpdateManager.h
  ../coUIContainer.h
  ../coUIElement.h
  ../coUIUserData.h
  ../coValuePoti.h
)

#SET(BASE_DEV_HEADERS
#)

ADD_COVISE_LIBRARY(coOpenVRUI SHARED ${BASE_SOURCES} ${BASE_HEADERS})
TARGET_LINK_LIBRARIES(coOpenVRUI ${COVISE_CONFIG_LIBRARY} ${COVISE_UTIL_LIBRARY})

COVISE_INSTALL_TARGET(coOpenVRUI)
