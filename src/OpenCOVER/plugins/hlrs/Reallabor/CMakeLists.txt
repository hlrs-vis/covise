set(HEADERS
  Reallabor.h
)
set(SOURCES
  Reallabor.cpp
)

# build the plugin "Template" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(Reallabor ${HEADERS} ${SOURCES})
