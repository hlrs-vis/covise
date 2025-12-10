using(osgcal)

set(HEADERS
    Traffic.h
    Geometry.h
    PedestrianGeometry.h
    CarGeometry.h
    sumo/ConnectorSumo.h
    sumo/socket.h
    sumo/storage.h
    sumo/TraCIAPI.h
    sumo/TraCIDefs.h
    sumo/TraCIConstants.h
    sumo/config.h
)
set(SOURCES
    Traffic.cpp
    Geometry.cpp
    PedestrianGeometry.cpp
    CarGeometry.cpp
    sumo/ConnectorSumo.cpp
    sumo/socket.cpp
    sumo/storage.cpp
    sumo/TraCIAPI.cpp
)

INCLUDE_DIRECTORIES(
   ${OSGCAL_INCLUDE_DIR}
)

cover_add_plugin(Traffic ${HEADERS} ${SOURCES})
target_link_libraries(Traffic 
    ${EXTRA_LIBS}
    ${OSGCAL_LIBRARIES}
)
