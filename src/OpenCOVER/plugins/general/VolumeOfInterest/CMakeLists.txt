#OpenCOVER plugins general VolumeOfInterest

SET(HEADERS
  VolumeOfInterestInteractor.h
  VolumeOfInterestPlugin.h
)

SET(SOURCES
  VolumeOfInterestInteractor.cpp
  VolumeOfInterestPlugin.cpp
)

cover_add_plugin(VolumeOfInterest)
