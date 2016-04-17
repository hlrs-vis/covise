set(HEADERS
  ReadTxt.h
)
set(SOURCES
  ReadTxt.cpp 
)

covise_add_module(Test TxtReader ${HEADERS} ${SOURCES})
TARGET_LINK_LIBRARIES(TxtReader coApi coAppl coCore ${EXTRA_LIBS})
