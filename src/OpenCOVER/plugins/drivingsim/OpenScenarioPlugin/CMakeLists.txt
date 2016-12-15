USING(XERCESC)

set(HEADERS
  OpenScenarioPlugin.h
)
set(SOURCES
  OpenScenarioPlugin.cpp
)

# build the plugin "OpenScenarioPlugin" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(OpenScenarioPlugin ${HEADERS} ${SOURCES})
