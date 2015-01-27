USE_VTK()
USING(BOOST)

include(FindProtobuf)
find_package(Protobuf)
find_package(ZeroMQ)
find_package(Eigen3)

SET(HEADERS
  CuttingTablet.h
  slicer.h
  networkzmq.h
  network.pb.h
)

SET(SOURCES
  CuttingTablet.cpp
  slicer.cpp
  networkzmq.cpp
  network.pb.cc
)

include_directories(${EIGEN3_INCLUDE_DIR} ${PROTOBUF_INCLUDE_DIR} ${ZEROMQ_INCLUDE_DIR})

#ADD_DEFINITIONS(-DLIC)

#SET(EXTRA_LIBS ${EXTRA_LIBS} protobuf zmq)

ADD_COVER_PLUGIN(CuttingTablet)
TARGET_LINK_LIBRARIES(CuttingTablet coOpenVRUI ${EXTRA_LIBS} ${PROTOBUF_LIBRARY} ${ZEROMQ_LIBRARY})
