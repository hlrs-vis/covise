# @file
# 
# CMakeLists.txt for sys vrb vrb (gui)
#
# @author Blasius Czink

USING(BOOST)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

INCLUDE_DIRECTORIES(
  ..
)

SET(VRB_SOURCES
  ../VRBServer.cpp
  ../VrbUiClientList.cpp
  ../VrbUiMessageHandler.cpp
  VRB.cpp
  VRBCurve.cpp
  VRBFileDialog.cpp
  VRBPopupMenu.cpp
  VRBapplication.cpp
  coRegister.cpp
  listVRBs.cpp
)

SET(VRB_MOC_HEADERS
  ../VRBServer.h
  ../VrbUiClientList.h
  ../VrbUiMessageHandler.h
  VRBCurve.h
  VRBFileDialog.h
  VRBPopupMenu.h
  VRBapplication.h
  coRegister.h
  listVRBs.h
)

SET(VRB_MOC_SRCS
  resources.qrc
)

ADD_COVISE_EXECUTABLE(vrb ${VRB_SOURCES} ${VRB_MOC_HEADERS} ${VRB_MOC_SRCS})
IF(BASEARCHSUFFIX STREQUAL "gecko" OR BASEARCHSUFFIX STREQUAL "heiner")
  ADD_COVISE_COMPILE_FLAGS(vrb "-Wno-non-virtual-dtor")
ENDIF()
# COVISE_WNOERROR(coVRML BASEARCH gecko heiner)

TARGET_LINK_LIBRARIES(vrb
                        coNet
                        coQtUtil
                        coUtil
                        coVRB
                        coVRBServer)
QT_USE_MODULES(vrb Core Gui Widgets)

COVISE_INSTALL_TARGET(vrb)

IF(UNIX)
    ADD_CUSTOM_COMMAND(TARGET vrb
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove -f ${COVISE_DESTDIR}/${COVISE_ARCHSUFFIX}/bin/vrbc
        COMMAND ${CMAKE_COMMAND} -E create_symlink vrb ${COVISE_DESTDIR}/${COVISE_ARCHSUFFIX}/bin/vrbc)
    INSTALL(FILES ${COVISE_DESTDIR}/${COVISE_ARCHSUFFIX}/bin/vrbc DESTINATION ${COVISE_ARCHSUFFIX}/bin)
ENDIF()
