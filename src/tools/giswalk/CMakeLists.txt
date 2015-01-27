cmake_minimum_required (VERSION 2.6)
#project (giswalk)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmakemodules;${CMAKE_MODULE_PATH}")

IF(WIN32)
  SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmakemodules/windows;${CMAKE_MODULE_PATH}")
ENDIF()

using(XercesC)
IF(XERCESC_FOUND)
  add_definitions(-DHAVE_XERCESC)
ENDIF(XERCESC_FOUND)
using(ZLIB)
using(JPEG)

using(TIFF)
IF(TIFF_FOUND)
  add_definitions(-DHAVE_TIFF)
ENDIF(TIFF_FOUND)


#add_library (libGiswalk SHARED gwTier.cpp gwTier.h  gwApp.cpp gwApp.h )
#target_link_libraries (libGiswalk ${EXTRA_LIBS} ) 

#set_target_properties(libGiswalk PROPERTIES DEFINE_SYMBOL  GWAPP_EXPORT )

#add_executable (giswalk giswalk.cpp giswalk.h)
#target_link_libraries (giswalk libGiswalk ${EXTRA_LIBS} ) 

ADD_COVISE_LIBRARY(libGiswalk SHARED gwTier.cpp gwTier.h  gwApp.cpp gwApp.h)
TARGET_LINK_LIBRARIES(libGiswalk ${EXTRA_LIBS})
set_target_properties(libGiswalk PROPERTIES DEFINE_SYMBOL  GWAPP_EXPORT )

COVISE_INSTALL_TARGET(libGiswalk)


ADD_COVISE_EXECUTABLE(giswalk giswalk.cpp giswalk.h)

TARGET_LINK_LIBRARIES(giswalk libGiswalk ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(giswalk)
