# @file
# 
# CMakeLists.txt for sys tabletUI
#
# @author Blasius Czink

SET(MAPEDITOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mapeditor")

INCLUDE_DIRECTORIES(
  "${COVISEDIR}/src/OpenCOVER/cover"
  "${MAPEDITOR_DIR}"
)

# include common tabletUI sources (sets TUI_SOURCES, TUI_HEADERS and TUI_MOC_HEADERS)
INCLUDE("tabletUI.cmake")

# append the rest
LIST(APPEND TUI_SOURCES tabletUI.cpp "${MAPEDITOR_DIR}/color/MEColorChooser.cpp")
LIST(APPEND TUI_MOC_HEADERS "${MAPEDITOR_DIR}/color/MEColorChooser.h")

QT_WRAP_CPP(TUI_MOC_SRCS ${TUI_MOC_HEADERS})

ADD_COVISE_EXECUTABLE(tabletUI MACOSX_BUNDLE ${TUI_SOURCES} ${TUI_HEADERS} ${TUI_MOC_HEADERS} ${TUI_MOC_SRCS})
qt_use_modules(tabletUI Xml UiTools Widgets)

if(APPLE)
   set_target_properties(tabletUI PROPERTIES OUTPUT_NAME "TabletUI")
endif(APPLE)

TARGET_LINK_LIBRARIES(tabletUI coNet coCore coUtil coConfig ${EXTRA_LIBS})
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_property(TARGET tabletUI PROPERTY WIN32_EXECUTABLE true)
endif()

COVISE_INSTALL_TARGET(tabletUI)
