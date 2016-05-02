set(HEADERS
  InteractorSensor.h
  StarCDPlugin.h
  StarRegion.h
  StarRegionInteractor.h
)
set(SOURCES
  InteractorSensor.cpp
  StarCDPlugin.cpp
  StarRegion.cpp
  StarRegionInteractor.cpp
)
cover_add_plugin(StarCDPlugin ${HEADERS} ${SOURCES})
set_target_properties(StarCDPlugin PROPERTIES OUTPUT_NAME StarCD)
