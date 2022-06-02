#OpenCOVER plugin to render trajectories for CapeReviso

ADD_DEFINITIONS()

SET(HEADERS
  TrajectoriesPlugin.h
)

SET(SOURCES
  TrajectoriesPlugin.cpp
)

cover_add_plugin(Trajectories)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_covise_compile_flags(CNC "-Wno-error=unused-result")
endif()
