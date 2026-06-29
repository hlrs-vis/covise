if(NOT COVISE_USE_QT)
    return()
endif()
USING(VRML)


SET(HEADERS
  ParticleViewer.h
  Particles.h
  VrmlNodeParticles.h
)
SET(SOURCES
  ParticleViewer.cpp
  Particles.cpp
  VrmlNodeParticles.cpp
)
cover_add_plugin(Particles)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_covise_compile_flags(Particles "-Wno-error=unused-result")
endif()
