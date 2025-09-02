set(SOURCES
LevenshteinDistane.cpp
)

set(HEADERS
export.h
LevenshteinDistane.h
)

covise_add_library(coStringUtil ${SOURCES} ${HEADERS})
COVISE_INSTALL_TARGET(coStringUtil)
COVISE_INSTALL_HEADERS(coStringUtil ${HEADERS})