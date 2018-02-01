USING(SISL)
SET(HEADERS
  NurbsSurface.h
  sisl_aux/aux2.h
  sisl_aux/sisl_aux.h
  sisl_aux/jonvec.h
)
SET(SOURCES
  sisl_aux/sisl_aux.cpp
  NurbsSurface.cpp
)
cover_add_plugin(NurbsSurface ${HEADERS} ${SOURCES})
