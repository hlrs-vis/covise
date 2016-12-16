
SET(HEADERS
  ARTracePlugin.h
)

SET(SOURCES
  ARTracePlugin.cpp  
)

if (COVISE_BUILD_ONLY_COVER_PREREQ)
    cover_add_plugin(ARInteractors)
else()
    cover_add_plugin(ARInteractors ${COVISE_APPL_LIBRARY} ${COVISE_SHM_LIBRARY})
endif()
