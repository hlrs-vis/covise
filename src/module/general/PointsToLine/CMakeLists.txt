set(HEADERS
    PointsToLine.h
)
set(SOURCES
    PointsToLine.cpp
)

covise_add_module(Mapper PointsToLine ${HEADERS} ${SOURCES})
covise_install_target(PointsToLine)
