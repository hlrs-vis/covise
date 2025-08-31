

USING(BOOST)
ADD_DEFINITIONS(-DMATERIAL)
#INCLUDE_DIRECTORIES()

# TODO: cmakeify use ADD_CUSTOM_COMMAND()

SET(HEADERS
  
)

SET(SOURCES
  ../Collect/Collect.cpp
)

SET(EXTRASOURCES
  ../Collect/Collect.h
#  file_generator.cpp
)

using(BOOST)

ADD_COVISE_MODULE(Tools Material ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Material  coApi coAppl coCore )

COVISE_INSTALL_TARGET(Material)
