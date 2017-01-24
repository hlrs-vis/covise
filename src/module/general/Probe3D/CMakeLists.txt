# ADD_DEFINITIONS()
INCLUDE_DIRECTORIES(
  ../Tracer
)

SET(HEADERS
  
)

SET(SOURCES
  Probe3D.cpp
  ../Tracer/BBoxAdmin.cpp
)

SET(EXTRASOURCES
  Probe3D.h
  ../Tracer/BBoxAdmin.h
)

ADD_COVISE_MODULE(Tools Probe3D ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Probe3D  coAlg coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(Probe3D)
