# PROJECT(comote)

USING(GLEW)

COVISE_FIND_PACKAGE(JPEGTURBO)

IF(NOT JPEGTURBO_FOUND)
  RETURN()
ENDIF()
if(NOT TURBOJPEG_HAVE_TJMCUWIDTH)
   return()
endif()

ADD_DEFINITIONS(-DUSE_TUIO=0 -DOSC_HOST_LITTLE_ENDIAN)

INCLUDE_DIRECTORIES(Tuio)
INCLUDE_DIRECTORIES(${JPEGTURBO_INCLUDE_DIR})

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

SET(HEADERS
    Debug.h
    FPSCounter.h
    MyTuioServer.h
    Shader.h
    RR/RR.h
    RR/RRCompressedTile.h
    RR/RRDecompressor.h
    RR/RRFrame.h
    RR/RRXEvent.h
    RR/tjplanar.h
)

QT_WRAP_CPP(MOC_SOURCES
    MainWidget.h
    MainWindow.h
    NetworkDialog.h
    RR/RRServer.h
)

SET(SOURCES
    Debug.cpp
    FPSCounter.cpp
    Main.cpp
    MainWidget.cpp
    MainWindow.cpp
    MyTuioServer.cpp
    NetworkDialog.cpp
    Shader.cpp
    RR/RRDecompressor.cpp
    RR/RRFrame.cpp
    RR/RRServer.cpp
    RR/tjplanar.c
)

QT_WRAP_UI(UI_SOURCES
    MainWindow.ui
    NetworkDialog.ui
)

QT_ADD_RESOURCES(RESOURCES
    comote.qrc
)

ADD_COVISE_EXECUTABLE(comote ${SOURCES} ${HEADERS} ${UI_SOURCES} ${RESOURCES} ${MOC_SOURCES})
qt_use_modules(comote OpenGL Widgets Network)

COVISE_INSTALL_TARGET(comote)

IF(UNIX OR MINGW)
  IF(NOT APPLE)
    ADD_COVISE_COMPILE_FLAGS(comote "-fopenmp")
    ADD_COVISE_LINK_FLAGS(comote "-fopenmp")
  ELSE()
    ADD_COVISE_COMPILE_FLAGS(comote "-Wno-deprecated")
  ENDIF()
ENDIF(UNIX OR MINGW)

IF(MSVC)
  ADD_COVISE_COMPILE_FLAGS(comote "/openmp")
  TARGET_LINK_LIBRARIES(comote winmm)
ENDIF(MSVC)

TARGET_LINK_LIBRARIES(comote
  ${GLEW_LIBRARY}
  ${TURBOJPEG_LIBRARIES}
  ${JPEGTURBO_LIBRARIES}
  ${OPENGL_LIBRARIES}
)
