# @file
# 
# CMakeLists.txt for kernel - net
#
# @author Blasius Czink

ADD_DEFINITIONS(-DNET_EXPORT)

if (OPENSSL_FOUND)
INCLUDE_DIRECTORIES(
  ${OPENSSL_INCLUDE_DIR}
)
add_definitions(-DHAVE_OPENSSL)
endif()

SET(NET_SOURCES
  covise_connect.cpp
  covise_host.cpp
  covise_socket.cpp
  dataHandle.cpp
  message.cpp
  message_sender_interface.cpp
  tokenbuffer.cpp
  tokenbuffer_serializer.cpp
  udpMessage.cpp
)

SET(NET_HEADERS
  covise_connect.h
  covise_host.h
  covise_socket.h
  dataHandle.h
  message.h
  message_macros.h
  message_sender_interface.h
  message_types.h
  tokenbuffer.h
  tokenbuffer_serializer.h
  tokenbuffer_util.h
  udpMessage.h
  udp_message_types.h

)

ADD_COVISE_LIBRARY(coNet ${COVISE_LIB_TYPE} ${NET_SOURCES} ${NET_HEADERS})
IF(MSVC)
ELSE(MSVC)
   ADD_COVISE_COMPILE_FLAGS(coNet "-Wno-deprecated-declarations")
ENDIF(MSVC)
TARGET_LINK_LIBRARIES(coNet coUtil coConfig ${ZLIB_LIBRARIES})
if (OPENSSL_FOUND)
TARGET_LINK_LIBRARIES(coNet ${OPENSSL_LIBRARIES})
endif()
IF(APPLE)
   COVISE_WNOERROR(coNet)
ENDIF()
COVISE_INSTALL_TARGET(coNet)
COVISE_INSTALL_HEADERS(net ${NET_HEADERS})
