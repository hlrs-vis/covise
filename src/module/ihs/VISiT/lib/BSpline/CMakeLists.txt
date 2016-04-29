SET(HEADERS
  include/bspline.h
)
SET(SOURCES
  bsplineknot.cpp
  bsplinenormal.cpp
  bsplinepoint.cpp
)
covise_add_library(visit_bspline STATIC ${EXTRASOURCES} ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES(visit_bspline)
COVISE_INSTALL_TARGET(visit_bspline)
