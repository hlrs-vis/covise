INCLUDE_DIRECTORIES(
../lib
)
ADD_DEFINITIONS("-DDRAFT_TUBE -DDEBUG -DDEBUG_ALL")

SET(HEADERS
  DraftTube.h
)
SET(SOURCES
  DraftTube.cpp
  InterpolateAreas.cpp
  CheckUserInput.cpp
  ../lib/General/4all/wg.cpp
  ../lib/General/4all/cg.cpp
  ../lib/General/4all/cg4c.cpp
)
covise_add_module(VISiT DraftTube ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(DraftTube coApi coAppl coCore visit_drafttube visit_bspline visit_general)
