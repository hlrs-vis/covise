USING(CG OPENGL)

SET(HEADERS
  ParticleViewer.h
  Particles.h
)
SET(SOURCES
  ParticleViewer.cpp
  Particles.cpp
)
cover_add_plugin(Particles)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_covise_compile_flags(Particles "-Wno-error=unused-result")
endif()
