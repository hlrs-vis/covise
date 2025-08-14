IF(NOT COVISE_USE_QT)
    RETURN()
ENDIF()

SET(HEADERS
CoviseConfigShader.h
)

SET(SOURCES
CoviseConfigShader.cpp
)
cover_add_plugin(CoviseConfigShader)
