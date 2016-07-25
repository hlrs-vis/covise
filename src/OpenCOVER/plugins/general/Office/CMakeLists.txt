
using(VRML)

SET(HEADERS
  OfficePlugin.h
  VrmlNodeOffice.h
)
SET(SOURCES
  OfficePlugin.cpp
  VrmlNodeOffice.cpp
)
cover_add_plugin(Office)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_covise_compile_flags(Office "-Wno-error=unused-result")
endif()
