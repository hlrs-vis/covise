# @file
# 
# CMakeLists.txt for kernel - dmgr
#
# @author Blasius Czink

ADD_DEFINITIONS(-DCOVISE_DMGR)

SET(DMGR_SOURCES
  dmgr_events.cpp
  dmgr_mem_avltrees.cpp
  dmgr_msg.cpp
  dmgr_pack_read.cpp
  dmgr_pack_write.cpp
  dmgr_process.cpp
  dmgr_shm.cpp
  dmgr_statics.cpp
)

SET(DMGR_HEADERS
  dmgr.h
  dmgr_mem_avltrees.h
  dmgr_packer.h
)

ADD_COVISE_LIBRARY(coDmgr ${COVISE_LIB_TYPE} ${DMGR_SOURCES} ${DMGR_HEADERS})

IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_COVISE_COMPILE_FLAGS(coDmgr "-fno-strict-aliasing")
ENDIF()

TARGET_LINK_LIBRARIES(coDmgr coDo coCore coConfig)

COVISE_INSTALL_TARGET(coDmgr)
COVISE_INSTALL_HEADERS(dmgr ${DMGR_HEADERS})
