#OpenCOVER plugins general SortLast

SET(HEADERS
  SortLast.h
  SortLastImplementation.h
  SortLastMaster.h
  SortLastSlave.h
)

SET(SOURCES
  SortLast.cpp
  SortLastImplementation.cpp
  SortLastMaster.cpp
  SortLastSlave.cpp
)

cover_add_plugin(SortLast)
