# @file
# 
# CMakeLists.txt for sys DevServer cereal (CEREALserver)
#
# @author Blasius Czink

if(NOT UNIX)
   return()
endif()

if(NOT OPENTHREADS_FOUND)
   return()
endif()

ADD_DEFINITIONS(-DSTANDALONE)

INCLUDE_DIRECTORIES(
  ${OPENTHREADS_INCLUDE_DIR}
  "${COVISEDIR}/src/OpenCOVER"
)

SET(CEREALSERVER_SOURCES
  CEREALserver.cpp
  "../../../legacy/bgLib.cpp"
)

SET(CEREALSERVER_HEADERS
  "../../../legacy/bgLib.h"
)

ADD_COVISE_EXECUTABLE(CEREALserver ${CEREALSERVER_SOURCES} ${CEREALSERVER_HEADERS})
TARGET_LINK_LIBRARIES(CEREALserver coDevServer) 

COVISE_INSTALL_TARGET(CEREALserver)
