#OpenCOVER plugins general PickSphere

SET(HEADERS
  PickSphereInteractor.h
  PickSpherePlugin.h
  SphereData.h
)

SET(SOURCES
  PickSphereInteractor.cpp
  PickSpherePlugin.cpp
)

cover_add_plugin(PickSphere ${COVISE_DO_LIBRARY} ${COVISE_SHM_LIBRARY})
