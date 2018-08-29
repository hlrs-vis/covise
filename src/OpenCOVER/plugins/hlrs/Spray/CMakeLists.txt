USING(Embree3)

SET(HEADERS
)
SET(SOURCES
gen.cpp
nozzle.cpp
nozzlemanager.cpp
Spray.cpp
nodevisitorvertex.cpp
)

cover_add_plugin(Spray ${HEADERS} ${SOURCES})

