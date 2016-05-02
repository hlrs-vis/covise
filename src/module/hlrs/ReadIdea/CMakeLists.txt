SET(HEADERS
)
SET(SOURCES
  ReadIdea.cpp
  dc.cpp
  memory.cpp
  sort.cpp
  edge.cpp
)
ADD_COVISE_MODULE(IO ReadIdea ${EXTRASOURCES})
covise_wnoerror(ReadIdea)
TARGET_LINK_LIBRARIES(ReadIdea coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadIdea)
