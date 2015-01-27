#OpenCOVER plugins general SortLastHPParComp

SET(HEADERS
  SortLastHPParComp.h
  SortLastImplementation.h
  SortLastMaster.h
  SortLastSlave.h
)

SET(SOURCES
  SortLastHPParComp.cpp
  SortLastImplementation.cpp
  SortLastMaster.cpp
  SortLastSlave.cpp
)

cover_add_plugin(SortLastHPParComp)
