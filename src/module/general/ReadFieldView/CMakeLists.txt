

SET(HEADERS
  ReadFieldView.h
)

SET(SOURCES
  ReadFieldView.cpp
)

SET(EXTRASOURCES
)


ADD_COVISE_MODULE(IO ReadFieldView ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ReadFieldView  coReader coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(ReadFieldView)
