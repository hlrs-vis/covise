USING(Fortran)

INCLUDE_DIRECTORIES(
../lib
)
SET(HEADERS
  DomainDecomposition.h
)
SET(SOURCES
  DomainDecomposition.cpp
)
covise_add_module(VISiT DomainDecomposition ${EXTRASOURCES} ${SOURCES} ${HEADERS})
target_link_libraries(DomainDecomposition coApi coAppl coCore visit_zerno visit_metis)
covise_install_target(DomainDecomposition)
