#OpenCOVER plugins general Measure

SET(HEADERS
  Dimension.h
  LinearDimension.h
  Measure.h
  Pin.h

)

SET(SOURCES
  Dimension.cpp
  LinearDimension.cpp
  Measure.cpp
  Pin.cpp
)

cover_add_plugin(Measure)
