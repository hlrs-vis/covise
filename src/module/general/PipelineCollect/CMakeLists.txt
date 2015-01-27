
SET(HEADERS
  
)

SET(SOURCES
  PipelineCollect.cpp
)

SET(EXTRASOURCES
  PipelineCollect.h
)

ADD_COVISE_MODULE(Tools PipelineCollect ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(PipelineCollect  coApi coAppl coCore coUtil)

COVISE_INSTALL_TARGET(PipelineCollect)
