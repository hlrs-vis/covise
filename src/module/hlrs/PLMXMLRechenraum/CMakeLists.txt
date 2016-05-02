if (NOT OPENSCENEGRAPH_FOUND)
   return()
endif()

USING(XERCESC)
#USING(BOOST)
SET(HEADERS
  PLMXMLRechenraum.h
  include/IOutil.h
  include/cov.h
  include/elements.h
  include/fatal.h
  include/flist.h
  include/ilist.h
  include/log.h
  include/model.h
  include/points.h
  Parser/PLMXMLParser.h
  Parser/FileReference.h
  Parser/functions.h
  Parser/NameId.h
)

INCLUDE_DIRECTORIES( 
${OPENSCENEGRAPH_INCLUDE_DIRS}
./Parser
)
SET(SOURCES
  cov.cpp
  elements.cpp
  fatal.cpp
  flist.cpp
  geogrid.cpp
  ilist.cpp
  log.cpp
  model.cpp
  points.cpp
  PLMXMLRechenraum.cpp
  IOutil.cpp 
  Parser/PLMXMLParser.cpp
  Parser/FileReference.cpp
  Parser/NameId.cpp
)
#ADD_SUBDIRECTORY(Parser)
ADD_COVISE_MODULE(HLRS PLMXMLRechenraum ${EXTRASOURCES})
covise_wnoerror(PLMXMLRechenraum)

IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(PLMXMLRechenraum "/Zc:wchar_t-")
ENDIF()

TARGET_LINK_LIBRARIES(PLMXMLRechenraum coApi coAppl coCore
${OPENSCENEGRAPH_LIBRARIES}
)

COVISE_INSTALL_TARGET(PLMXMLRechenraum)
