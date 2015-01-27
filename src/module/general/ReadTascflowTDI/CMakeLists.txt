return()

#COVISE_AVAILABLE_ON(BASEARCH sgin32 sgi64)
#ADD_DEFINITIONS()
#INCLUDE_DIRECTORIES()

SET(HEADERS
  
)

SET(SOURCES
  BcfFile.cpp
  FieldFile.cpp
  ReadTascflowTDI.cpp
)

SET(EXTRASOURCES
  BcfFile.h
  FieldFile.h
  ReadTascflowTDI.h
  blocko.f
  dims.f
  file.f
  gnum.f
  nexts.f
  ngbl.f
  nxtr.f
  rclose.f
  region.f
  rinit.f
  rread.f
  scalar.f
  subreg.f
)

ADD_COVISE_MODULE(IO ReadTascflowTDI ${EXTRASOURCES} )
# old CONFIG: qt warn_on release incremental link_prl colib coapi coappl f77 math
# old LIBS: $$LIBDIR/tdi.a $$LIBDIR/aclgraph.a $$LIBDIR/tasclib.a $$LIBDIR/libz.a $$LIBDIR/system.a
# old links: 
TARGET_LINK_LIBRARIES(ReadTascflowTDI  coApi coAppl coCore )

COVISE_INSTALL_TARGET(ReadTascflowTDI)
