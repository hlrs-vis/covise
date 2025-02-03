# @file
# 
# CMakeLists.txt for kernel - tui
#
# @author Blasius Czink

USING(Virvo:transfunc optional) # for transfer function editor
USING(WiringPi optional)

set(CMAKE_AUTORCC ON)

ADD_DEFINITIONS(-DCOVISE_TUI)

# set TUI_SOURCES, TUI_HEADERS and TUI_MOC_HEADERS


SET(TUI_SOURCES
  mapviewer.qrc
  qtcolortriangle.cpp
  qtpropertyDialog.cpp
  TUIColorTriangle.cpp
  TUIColorButton.cpp
  TUIContainer.cpp
  TUITab.cpp
  TUITabFolder.cpp
  TUIProgressBar.cpp
  TUIButton.cpp
  TUIToggleButton.cpp
  TUIToggleBitmapButton.cpp
  TUIFrame.cpp
  TUIGroupBox.cpp
  TUIScrollArea.cpp
  TUISplitter.cpp
  TUIComboBox.cpp
  TUIListBox.cpp
  TUIFloatSlider.cpp
  TUISlider.cpp
  TUIFloatEdit.cpp
  TUIIntEdit.cpp
  TUILabel.cpp
  TUIElement.cpp
  TUILineEdit.cpp
  TUILineCheck.cpp
  # TUITextSpinEdit.cpp
  TUIMain.cpp
  #TUIMainFrame.cpp
  TUIMainWindow.cpp
  TUINavElement.cpp
  TUISGBrowserTab.cpp
  TUIColorTab.cpp
  TUIMap.cpp
  TUIEarthMap.cpp
  TUIFileBrowserButton.cpp
  FileBrowser/FileBrowser.cpp
  TUITextCheck.cpp
  TUITextEdit.cpp
  TUIAnnotationTab.cpp
  TUIColorWidget.cpp
  TUIPopUp.cpp
  TUIUITab.cpp
  TUIUI/TUIUIWidget.cpp
  TUIUI/TUIUIWidgetSet.cpp
  TUIUI/TUIUIScriptWidget.cpp
)

SET(TUI_HEADERS
  export.h
  TUIContainer.h
  TUIElement.h
  TUIFrame.h
  TUIGroupBox.h
  TUILabel.h
  TUIMain.h
  TUINavElement.h
  TUISplitter.h
  TUITab.h
  FileBrowser/RemoteClientDialog.h
)

SET(TUI_MOC_HEADERS
  qtcolortriangle.h
  qtpropertyDialog.h
  TUIColorTriangle.h
  TUIColorButton.h
  TUISGBrowserTab.h
  TUIColorTab.h
  TUIButton.h
  TUIComboBox.h
  TUIIntEdit.h
  TUIProgressBar.h
  TUIFloatEdit.h
  TUIFloatSlider.h
  TUISlider.h
  TUILineEdit.h
  TUILineCheck.h
  TUIListBox.h
  TUIEarthMap.h
  TUIMap.h
  #TUIMainFrame.h
  TUIMainWindow.h
  TUITabFolder.h
  # TUITextSpinEdit.h
  TUIToggleButton.h
  TUIToggleBitmapButton.h
  TUIFileBrowserButton.h
  FileBrowser/FileBrowser.h
  TUITextCheck.h
  TUITextEdit.h
  TUIAnnotationTab.h
  TUIColorWidget.h
  TUIPopUp.h
  TUIUITab.h
  TUIUI/TUIUIWidget.h
  TUIUI/TUIUIWidgetSet.h
  TUIUI/TUIUIScriptWidget.h
)

if (WIRINGPI_FOUND)
   SET(TUI_SOURCES ${TUI_SOURCES}
      Thyssen.cpp
      ThyssenButton.cpp
      )
   SET(TUI_HEADERS ${TUI_HEADERS}
      Thyssen.h
      ThyssenButton.h
      )
   SET(TUI_MOC_HEADERS ${TUI_MOC_HEADERS}
      ThyssenButton.h
      )
endif (WIRINGPI_FOUND)

if (COVISE_USE_VIRVO)
   SET(TUI_SOURCES ${TUI_SOURCES}
      TUIFunctionEditorTab.cpp
      TUITFEWidgets.cpp
      TUITFEditor.cpp
      TUITF2DEditor.cpp
      )
   SET(TUI_HEADERS ${TUI_HEADERS}
      TUITFEWidgets.h
      )
   SET(TUI_MOC_HEADERS ${TUI_MOC_HEADERS}
      TUIFunctionEditorTab.h
      TUITFEditor.h
      TUITF2DEditor.h
      )
endif()


if (COVISE_USE_WEBENGINE)
   SET(TUI_MOC_HEADERS ${TUI_MOC_HEADERS}
     TUIWebview.h
   )
   SET(TUI_SOURCES ${TUI_SOURCES}
     TUIWebview.cpp
   )
endif()

qt_wrap_cpp(TUI_MOC_SRCS ${TUI_MOC_HEADERS})

SET(COTUI_SOURCES ${TUI_SOURCES})
SET(COTUI_HEADERS  coAbstractTabletUI.h ${TUI_HEADERS})

COVISE_ADD_LIBRARY(coTUI ${COTUI_HEADERS} ${COTUI_SOURCES} ${TUI_MOC_HEADERS} ${TUI_MOC_SRCS})

qt_use_modules(coTUI Xml UiTools Widgets Quick)

if (COVISE_USE_WEBENGINE)
    qt_use_modules(coTUI WebEngine WebEngineWidgets)
  ADD_DEFINITIONS(-DUSE_WEBENGINE)
endif()


if(Qt5Positioning_FOUND)
    qt_use_modules(coTUI Positioning Location)
endif(Qt5Positioning_FOUND)

if(NOT WIN32)
    add_covise_compile_flags(coTUI "-Wno-error=deprecated-declarations")
endif(NOT WIN32)

TARGET_LINK_LIBRARIES(coTUI coNet coUtil coConfig coQtUtil ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(coTUI)
COVISE_INSTALL_HEADERS(tui ${COTUI_HEADERS})
