
# CONFIG *= wnouninitialized # because of IsoSurfaceGPMUtil.h

SET(HEADERS
  ShiftValues.h
  ShiftValuesPoint.h
)

SET(SOURCES
  ShiftValuesPoint.cpp
  ShiftValues.cpp
)

SET(EXTRASOURCES
  ShiftValues.h
  ShiftValuesPoint.h
)

ADD_COVISE_MODULE(Mapper ShiftValues )
TARGET_LINK_LIBRARIES(ShiftValues  coAlg coApi coAppl coCore )

COVISE_INSTALL_TARGET(ShiftValues)
