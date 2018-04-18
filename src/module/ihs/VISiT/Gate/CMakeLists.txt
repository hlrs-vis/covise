INCLUDE_DIRECTORIES(
../lib
)

SET(HEADERS
  Gate.h
)
SET(SOURCES
  Gate.cpp
  ModLib.cpp
  ../lib/General/4all/wg.cpp
  ../lib/General/4all/cg.cpp
  ../lib/General/4all/cg4c.cpp
)

ADD_DEFINITIONS(-DGATE -DCONTOURS)

covise_add_module(VISiT Gate ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(Gate coApi coAppl coCore visit_gate visit_general visit_bspline)
covise_install_target(Gate)
