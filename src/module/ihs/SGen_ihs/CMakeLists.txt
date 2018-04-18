using(Fortran)

SET(HEADERS
  SGen_ihs.h
)
SET(SOURCES
  SGen_ihs.cpp
  ausgabe.f
  axnet.f
  eingabe.f
  geogen.f
  netzlesen.f
  profil.f
  rechnen.f
)

covise_add_module(IO SGen_ihs ${SOURCES} ${HEADERS})
covise_wnoerror(SGen_ihs)
target_link_libraries(SGen_ihs coApi coAppl coCore)
covise_install_target(SGen_ihs)
