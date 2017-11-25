# @file
# 
# CMakeLists.txt for kernel - tui
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCOVISE_TUI)

SET(COTUI_SOURCES)
SET(COTUI_HEADERS  coAbstractTabletUI.h)

COVISE_INSTALL_HEADERS(tui ${COTUI_HEADERS})
