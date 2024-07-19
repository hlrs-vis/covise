#OpenCOVER plugins general PBufferSnapShot

if(NOT COVISE_USE_QT)
    return()
endif()

SET(HEADERS
  PBufferSnapShot.h
)

SET(SOURCES
  PBufferSnapShot.cpp
)

cover_add_plugin(PBufferSnapShot)
