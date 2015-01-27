SET(HEADERS
)
SET(SOURCES
  weather.cpp
)
ADD_COVISE_MODULE(Simulation weather ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(weather coApi coAppl coCore)
COVISE_INSTALL_TARGET(weather)
