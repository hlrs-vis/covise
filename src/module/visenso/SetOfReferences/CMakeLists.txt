set(HEADERS
)

set(SOURCES
   SetOfReferences.cpp
)

set(EXTRASOURCES
   SetOfReferences.h
)

add_covise_module(Tools SetOfReferences ${EXTRASOURCES} )
target_link_libraries(SetOfReferences  coApi coAppl coCore )
covise_install_target(SetOfReferences)
