SET(HEADERS
  include/diagonal.h
  include/radial.h
  include/rr2cov.h
  include/rr_dbpreset.h
)
SET(SOURCES
  dr_createmerid.cpp
  rr_angles.cpp
  rr_comp.cpp
  rr_createmerid.cpp
  rr_initbe.cpp
  rr_lsurfbladeelem.cpp
  main.cpp
  rr_bemerintersect.cpp
  rr_conduitareas.cpp
  rr_csurfbladelem2.cpp
  rr_interpolbp.cpp
  rr_modifybe4cov.cpp
  rr2cov.cpp
  rr_bladeedges.cpp
  rr_contourdata.cpp
  rr_csurfbladelem.cpp
  rr_io.cpp
  rr_msurfbladelem.cpp
)
ADD_DEFINITIONS("-DRADIAL_RUNNER -DCOVISE_MODULE")

covise_add_library(visit_radialrunner STATIC ${EXTRASOURCES} ${SOURCES}
   ${HEADERS})
TARGET_LINK_LIBRARIES(visit_radialrunner)
COVISE_INSTALL_TARGET(visit_radialrunner)
