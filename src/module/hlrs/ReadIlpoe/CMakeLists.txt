set(HEADERS
  ReadIlpoe.h
)
set(SOURCES
  ReadIlpoe.cpp 
)

covise_add_module(IO ReadIlpoe ${HEADERS} ${SOURCES})
covise_wnoerror(ReadIlpoe)
TARGET_LINK_LIBRARIES(ReadIlpoe coApi coAppl coCore ${EXTRA_LIBS})
covise_install_target(ReadIlpoe)
