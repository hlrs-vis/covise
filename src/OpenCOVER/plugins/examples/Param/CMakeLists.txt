if(NOT COVISE_USE_QT)
    return()
endif()

SET(HEADERS
  ParamPlugin.h
)
SET(SOURCES
  ParamPlugin.cpp
)
cover_add_plugin(Param ${HEADERS} ${SOURCES})
