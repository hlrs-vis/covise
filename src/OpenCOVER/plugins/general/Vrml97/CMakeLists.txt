# OpenCOVER plugins general Vrml97
USING(VTK optional)
ADD_DEFINITIONS(-DVRML97_IMPORT_PLUGIN -DNEW_OSG)

IF(NOT (${OPENSCENEGRAPH_VERSION} VERSION_LESS "2.9"))
  SET(SDL_BUILDING_LIBRARY ON)
  COVISE_FIND_PACKAGE(SDL)
  IF(SDL_FOUND)
    INCLUDE_DIRECTORIES(${SDL_INCLUDE_DIR})
    SET(EXTRA_LIBS ${SDL_LIBRARY})
  ENDIF()
ENDIF()

# INCLUDE_DIRECTORIES( )

SET(LIB_HEADERS
  coSensiveSensor.h
  ViewerObject.h
  ViewerOsg.h
  VrmlNodeMirrorCamera.h
  VrmlNodeARSensor.h
  VrmlNodeTUI.h
  VrmlNodeTimesteps.h
  VrmlNodeMultiTouchSensor.h
  VrmlNodeCOVISEObject.h
  VrmlNodeClippingPlane.h
)

SET(LIB_SOURCES
  coSensiveSensor.cpp
  ViewerObject.cpp
  ViewerOsg.cpp
  VrmlNodeMirrorCamera.cpp
  VrmlNodeARSensor.cpp
  VrmlNodeTUI.cpp
  VrmlNodeTimesteps.cpp
  VrmlNodeMultiTouchSensor.cpp
  VrmlNodeCOVISEObject.cpp
  VrmlNodeClippingPlane.cpp
)

add_covise_library(Vrml97Cover SHARED ${LIB_SOURCES})
target_link_libraries(Vrml97Cover coOpenPluginUtil coOpenCOVER coOpenVRUI
   coOSGVRUI ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_CONFIG_LIBRARY}
   ${COVISE_UTIL_LIBRARY} ${COVISE_GRMSG_LIBRARY} coVRML ${OPENSCENEGRAPH_LIBRARIES}) # ${CMAKE_THREAD_LIBS_INIT})
qt_use_modules(Vrml97Cover Core)
covise_install_target(Vrml97Cover)

set(HEADERS
  ListenerCover.h
  SystemCover.h
  Vrml97Plugin.h
)

set(SOURCES
  ListenerCover.cpp
  SystemCover.cpp
  Vrml97Plugin.cpp
)

IF(NOT (${OPENSCENEGRAPH_VERSION} VERSION_LESS "2.9"))
  IF(SDL_FOUND)
    LIST(APPEND HEADERS SDLAudio.h)
    LIST(APPEND SOURCES SDLAudio.cpp)
  ENDIF()
ENDIF()

cover_add_plugin(Vrml97)
COVISE_WNOERROR(Vrml97 BASEARCH intrepid tuatara)
TARGET_LINK_LIBRARIES(Vrml97 Vrml97Cover coVRML ${COVISE_GRMSG_LIBRARY} ${EXTRA_LIBS})
if(NOT MSVC)
   add_covise_compile_flags(Vrml97 "-Wno-deprecated-declarations")
endif()

