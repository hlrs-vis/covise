# @file
# 
# CMakeLists.txt for sys tabletUI
#
# @author Blasius Czink
if(NOT COVISE_USE_QT)
    return()
endif()

ADD_COVISE_EXECUTABLE(tabletUI MACOSX_BUNDLE tabletUI.cpp)
target_link_libraries(tabletUI coTUI)
qt_use_modules(tabletUI Xml UiTools Widgets Quick)

if(APPLE)
   set_target_properties(tabletUI PROPERTIES OUTPUT_NAME "TabletUI")
endif(APPLE)

TARGET_LINK_LIBRARIES(tabletUI coNet coUtil coQtUtil coConfig ${EXTRA_LIBS})
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_property(TARGET tabletUI PROPERTY WIN32_EXECUTABLE true)
endif()

COVISE_INSTALL_TARGET(tabletUI)
