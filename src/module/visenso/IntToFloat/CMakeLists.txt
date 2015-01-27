set(HEADERS
)

set(SOURCES
   IntToFloat.cpp
)

set(EXTRASOURCES
   IntToFloat.h
)

add_covise_module(Converter IntToFloat ${EXTRASOURCES} )
target_link_libraries(IntToFloat  coApi coAppl coCore )
covise_install_target(IntToFloat)
