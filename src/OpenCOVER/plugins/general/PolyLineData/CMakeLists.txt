#OpenCOVER plugins general PolyLineData
if(NOT COVISE_USE_QT)
    return()
endif()

SET(HEADERS
  PolyLineData.h
)

SET(SOURCES
  PolyLineData.cpp
)

cover_add_plugin(PolyLineData)
