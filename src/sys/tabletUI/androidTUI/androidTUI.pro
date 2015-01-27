#-------------------------------------------------
#
# Project created by QtCreator 2014-11-01T21:15:40
#
#-------------------------------------------------

QT       += core gui uitools xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = androidTUI
TEMPLATE = app
DEFINES += ANDROID_TUI
DEFINES += BYTESWAP
INCLUDEPATH += ../../../kernel
INCLUDEPATH += ..
INCLUDEPATH += ../../mapeditor
INCLUDEPATH += ../../../renderer/OpenCOVER/kernel

SOURCES +=\
    ../qtcolortriangle.cpp \
    ../qtpropertyDialog.cpp \
    ../tabletUI.cpp \
    ../TUIAnnotationTab.cpp \
    ../TUIApplication.cpp \
    ../TUIButton.cpp \
    ../TUIColorButton.cpp \
    ../TUIColorTab.cpp \
    ../TUIColorTriangle.cpp \
    ../TUIColorWidget.cpp \
    ../TUIComboBox.cpp \
    ../TUIContainer.cpp \
    ../TUIElement.cpp \
    ../TUIFileBrowserButton.cpp \
    ../TUIFloatEdit.cpp \
    ../TUIFloatSlider.cpp \
    ../TUIFrame.cpp \
    ../TUIIntEdit.cpp \
    ../TUILabel.cpp \
    ../TUILineCheck.cpp \
    ../TUILineEdit.cpp \
    ../TUIListBox.cpp \
    ../TUIMap.cpp \
    ../TUINavElement.cpp \
    ../TUIPopUp.cpp \
    ../TUIProgressBar.cpp \
    ../TUIScrollArea.cpp \
    ../TUISGBrowserTab.cpp \
    ../TUISlider.cpp \
    ../TUISplitter.cpp \
    ../TUITab.cpp \
    ../TUITabFolder.cpp \
    ../TUITextCheck.cpp \
    ../TUITextEdit.cpp \
    ../TUITextureTab.cpp \
    ../TUIToggleBitmapButton.cpp \
    ../TUIToggleButton.cpp \
    ../TUIUITab.cpp \
    ../wce_connect.cpp \
    ../wce_host.cpp \
    ../wce_msg.cpp \
    ../wce_Restraint.cpp \
    ../wce_socket.cpp \
    ../FileBrowser/FileBrowser.cpp \
    ../../mapeditor/color/MEColorChooser.cpp \
    ../TUIUI/TUIUIScriptWidget.cpp \
    ../TUIUI/TUIUIWidget.cpp \
    ../TUIUI/TUIUIWidgetSet.cpp

HEADERS  += \
    ../qtcolortriangle.h \
    ../qtpropertyDialog.h \
    ../TUIAnnotationTab.h \
    ../TUIApplication.h \
    ../TUIButton.h \
    ../TUIColorButton.h \
    ../TUIColorTab.h \
    ../TUIColorTriangle.h \
    ../TUIColorWidget.h \
    ../TUIComboBox.h \
    ../TUIContainer.h \
    ../TUIElement.h \
    ../TUIFileBrowserButton.h \
    ../TUIFloatEdit.h \
    ../TUIFloatSlider.h \
    ../TUIFrame.h \
    ../TUIIntEdit.h \
    ../TUILabel.h \
    ../TUILineCheck.h \
    ../TUILineEdit.h \
    ../TUIListBox.h \
    ../TUIMap.h \
    ../TUINavElement.h \
    ../TUIPopUp.h \
    ../TUIProgressBar.h \
    ../TUIScrollArea.h \
    ../TUISGBrowserTab.h \
    ../TUISlider.h \
    ../TUISplitter.h \
    ../TUITab.h \
    ../TUITabFolder.h \
    ../TUITextCheck.h \
    ../TUITextEdit.h \
    ../TUITextureTab.h \
    ../TUIToggleBitmapButton.h \
    ../TUIToggleButton.h \
    ../TUIUITab.h \
    ../wce_connect.h \
    ../wce_host.h \
    ../wce_msg.h \
    ../wce_Restraint.h \
    ../wce_socket.h \
    ../FileBrowser/FileBrowser.h \
    ../FileBrowser/RemoteClientDialog.h \
    ../TUIUI/TUIUIScriptWidget.h \
    ../TUIUI/TUIUIWidget.h \
    ../TUIUI/TUIUIWidgetSet.h \
    ../../mapeditor/color/MEColorChooser.h

CONFIG += mobility
MOBILITY = 

FORMS += \
    ../FileBrowser/ClientDialog.ui

OTHER_FILES += \
    android/AndroidManifest.xml

RESOURCES += \
    tui.qrc

ANDROID_PACKAGE_SOURCE_DIR = $$PWD/android

