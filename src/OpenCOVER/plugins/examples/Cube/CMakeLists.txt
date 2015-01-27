SET(HEADERS
)
SET(SOURCES
  CubeInteractor.cpp
  CubePlugin.cpp
)
cover_add_plugin(CubePlugin ${HEADERS} ${SOURCES})
SET_TARGET_PROPERTIES(CubePlugin PROPERTIES OUTPUT_NAME Cube)
