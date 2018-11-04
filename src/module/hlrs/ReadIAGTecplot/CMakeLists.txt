
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
covise_install_target(ReadIAGTecplot)
if(NOT MSVC)
    add_covise_compile_flags(ReadIAGTecplot "-Wno-error=deprecated-declarations")
endif()
target_link_libraries(ReadIAGTecplot coReader ${EXTRA_LIBS})
