set(SOURCES)
set(HEADERS export.h threadworker.h)

ADD_COVISE_LIBRARY(coThreadUtil ${SOURCES} ${HEADERS})
COVISE_INSTALL_TARGET(coThreadUtil)
COVISE_INSTALL_HEADERS(coThreadUtil ${HEADERS})
COVISE_WNOERROR(coThreadUtil BASEARCH zebu)
set_target_properties(coThreadUtil PROPERTIES LINKER_LANGUAGE CXX)
