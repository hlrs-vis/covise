# @file
# 
# CMakeLists.txt for kernel - shm
#
# @author Blasius Czink

ADD_DEFINITIONS(-DSHM_EXPORTS)

SET(SHM_SOURCES
  covise_shm.cpp
  covise_shmalloc.cpp
)

SET(SHM_HEADERS
  covise_shm.h
)

SET(EXTRA_LIBS "")
IF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
   SET(EXTRA_LIBS ${EXTRA_LIBS} -lrt)
ENDIF()

ADD_COVISE_LIBRARY(coShm ${COVISE_LIB_TYPE} ${SHM_SOURCES} ${SHM_HEADERS})
TARGET_LINK_LIBRARIES(coShm coUtil coConfig coNet ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(coShm)
COVISE_INSTALL_HEADERS(shm ${SHM_HEADERS})
