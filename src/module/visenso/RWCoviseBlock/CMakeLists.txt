set(HEADERS
)

set(SOURCES
   LinesObject.cpp
   OutputObject.cpp
   OutputObjectFactory.cpp
   PolygonObject.cpp
   RWCoviseBlock.cpp
   ScalDataObject.cpp
   SetObject.cpp
)

set(EXTRASOURCES
   LinesObject.h
   OutputObjectFactory.h
   OutputObject.h
   PolygonObject.h
   RWCoviseBlock.h
   ScalDataObject.h
   SetObject.h
)

add_covise_module(Converter RWCoviseBlock ${EXTRASOURCES} )
target_link_libraries(RWCoviseBlock  coApi coAppl coCore )
covise_install_target(RWCoviseBlock)
