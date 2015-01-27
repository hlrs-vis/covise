# @file
# 
# CMakeLists.txt for config - coEditor
#
# @author Blasius Czink

INCLUDE_DIRECTORIES(
  "${XERCESC_INCLUDE_DIR}"
)

SET(COEDITOR_SOURCES
  main.cpp
  coEditorMainWindow.cpp
  coEditorEntryWidget.cpp
  coEditorBoolValueWidget.cpp
  coEditorTextValueWidget.cpp
  coEditorValidatedQLineEdit.cpp
  ../kernel/coConfigEntryToEditor.cpp
  coEditorGroupWidget.cpp
)

SET(COEDITOR_HEADERS
  ../coConfigEditorController.h
  ../coConfigEntryToEditor.h
  ../kernel/coConfigSchema.h
)

SET(COEDITOR_MOC_HEADERS
  coEditorMainWindow.h
  coEditorEntryWidget.h
  coEditorValueWidget.h
  coEditorBoolValueWidget.h
  coEditorTextValueWidget.h
  coEditorValidatedQLineEdit.h
  coEditorGroupWidget.h
)

QT_ADD_RESOURCES(COEDITOR_RCC_FILES application.qrc)
QT_WRAP_CPP(COEDITOR_MOC_SRCS ${COEDITOR_MOC_HEADERS})

ADD_COVISE_EXECUTABLE(coEditor MACOSX_BUNDLE ${COEDITOR_SOURCES} ${COEDITOR_HEADERS} ${COEDITOR_MOC_SRCS} ${COEDITOR_RCC_FILES})
TARGET_LINK_LIBRARIES(coEditor coConfig) 
qt_use_modules(coEditor Core Xml Widgets)

if(APPLE)
   set_target_properties(coEditor PROPERTIES OUTPUT_NAME "COVISE Config Editor")
endif()

COVISE_WNOERROR(coEditor)
COVISE_INSTALL_TARGET(coEditor)
