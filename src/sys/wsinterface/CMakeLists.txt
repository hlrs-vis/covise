# @file
# 
# CMakeLists.txt for sys - wsinterface
#
# @author Blasius Czink

USING(GSOAP)

IF(NOT COVISE_BUILD_WEBSERVICE)
  RETURN()
ENDIF()

ADD_DEFINITIONS(-DWITH_NONAMESPACES)

QT_ADD_RESOURCES(WSI_RESOURCE_FILES "../../../share/covise/web/web.qrc" OPTIONS "-no-compress")

INCLUDE_DIRECTORIES(
  ${GSOAP_INCLUDE_DIR}
  "${COVISEDIR}/src/kernel/wslib"
  "${COVISEDIR}/${ARCHSUFFIX}/include"
  "${CMAKE_BINARY_DIR}/src/kernel/wslib"
)

SET(WSI_SOURCES
  WSMessageHandler.cpp
  WSCOVISEService.cpp
  WSServer.cpp
  WSMain.cpp
  WSMainHandler.cpp
  WSEventManager.cpp
)

SET(WSI_HEADERS
  WSCOVISEService.h
  WSServer.h
  WSEventManager.h
)

SET(WSI_MOC_HEADERS
  WSMessageHandler.h
  WSMainHandler.h
)

QT_WRAP_CPP(WSI_MOC_SRCS ${WSI_MOC_HEADERS})

ADD_COVISE_EXECUTABLE(wsinterface MACOSX_BUNDLE ${WSI_SOURCES} ${WSI_HEADERS} ${WSI_MOC_HEADERS} ${WSI_MOC_SRCS} ${WSI_RESOURCE_FILES})
qt_use_modules(wsinterface Xml Network Gui Widgets)
COVISE_WNOERROR(wsinterface)
TARGET_LINK_LIBRARIES(wsinterface coWS coDo coCore coConfig) # ${CMAKE_THREAD_LIBS_INIT}

COVISE_INSTALL_TARGET(wsinterface)
