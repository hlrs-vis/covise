#OpenCOVER plugins general CNC

USING(CG)
ADD_DEFINITIONS(-DAA -DBB -DCC)

SET(HEADERS
  CNCPlugin.h
  rs274ngc_return.hh
  rs274ngc.hh
  canon.hh
)

SET(SOURCES
  CNCPlugin.cpp
  canon_pre.cpp
  rs274ngc_pre.cpp
)

cover_add_plugin(CNC)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_covise_compile_flags(CNC "-Wno-error=unused-result")
endif()
