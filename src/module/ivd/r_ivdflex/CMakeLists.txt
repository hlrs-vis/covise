SET(HEADERS
)
SET(SOURCES
  r_ivdflex.cpp
)
ADD_COVISE_MODULE(IO r_ivdflex ${EXTRASOURCES})
covise_wnoerror(r_ivdflex)
TARGET_LINK_LIBRARIES(r_ivdflex coApi coAppl coCore)
COVISE_INSTALL_TARGET(r_ivdflex)
