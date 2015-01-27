set(HEADERS
)

set(SOURCES
   MixColors.cpp
)

set(EXTRASOURCES
   MixColors.h
)

add_covise_module(Tools MixColors ${EXTRASOURCES} )
target_link_libraries(MixColors  coApi coAppl coCore )
covise_install_target(MixColors)
