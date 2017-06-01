
set(HEADERS
   ReadIAGTecplot.h
   mesh.h
   tecplotfile.h
   sources.h
   topo.h
)
set(SOURCES
   ReadIAGTecplot.cpp
   mesh.cpp
   tecplotfile.cpp
)

covise_add_module(IO ReadIAGTecplot ${HEADERS} ${SOURCES})
TARGET_LINK_LIBRARIES(ReadIAGTecplot coReader ${EXTRA_LIBS})
add_covise_compile_flags(ReadIAGTecplot "-Wno-error=deprecated-declarations")
