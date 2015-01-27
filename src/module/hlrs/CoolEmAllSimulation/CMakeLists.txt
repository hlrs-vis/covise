SET(HEADERS
  CoolEmAllSimulation.h
)
SET(SOURCES
  CoolEmAllSimulation.cpp
)
ADD_COVISE_MODULE(Simulation CoolEmAllSimulation ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(CoolEmAllSimulation coApi coAppl coCore)
COVISE_INSTALL_TARGET(CoolEmAllSimulation)
