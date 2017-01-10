USING(Boost)

SET(HEADERS
  
)

SET(SOURCES
  ParticleTrails.cpp
)

SET(EXTRASOURCES
  ParticleTrails.h
)

ADD_COVISE_MODULE(Tools ParticleTrails ${EXTRASOURCES} )

TARGET_LINK_LIBRARIES(ParticleTrails  coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(ParticleTrails)
