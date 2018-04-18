USING(CGNS)

IF (CGNS_FOUND)
  MESSAGE(STATUS "IHS::compiling writeToCgns")
ELSE (CGNS_FOUND)
  MESSAGE("IHS::skipping writeToCgns, because of missing libraries")
  MESSAGE("IHS::CGNS_FOUND=${CGNS_FOUND}")
  RETURN()
ENDIF (CGNS_FOUND)

SET(HEADERS
  writeToCgns.h
)
SET(SOURCES
  writeToCgns.cpp
)
covise_add_module(IO writeToCgns ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(writeToCgns coApi coAppl coCore ${EXTRA_LIBS})
covise_install_target(writeToCgns)
