IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(XERCESC OSGCAL CAL3D)
using(VRML)
using(V8)
using(VehicleUtil)

covise_find_package(OpenSceneGraph 3.2.0 COMPONENTS osgTerrain)
COVISE_FIND_PACKAGE(osgTerrain)
COVISE_FIND_PACKAGE(GDAL)

IF(NOT OSGCAL_FOUND)
  RETURN()
ENDIF()

if(NOT GDAL_FOUND)
    return()
endif()

SET(HEADERS
  coTrafficSimulation.h
  TrafficSimulation.h
  Vehicle.h
  DonkeyCar.h
  VehicleGeometry.h
  CarGeometry.h
  VehicleManager.h
  VehicleFactory.h
  HumanVehicle.h
  HumanVehicleGeometry.h
  AgentVehicle.h
  VehicleUtils.h
  PedestrianManager.h
  Pedestrian.h
  PedestrianGeometry.h
  PedestrianFactory.h
  PorscheFFZ.h
  UDPBroadcast.h
  projects/radarcones.hpp
)
SET(SOURCES
  coTrafficSimulation.cpp
  TrafficSimulation.cpp
  Vehicle.cpp
  DonkeyCar.cpp
  CarGeometry.cpp
  VehicleManager.cpp
  VehicleFactory.cpp
  HumanVehicle.cpp
  HumanVehicleGeometry.cpp
  AgentVehicle.cpp
  VehicleUtils.cpp
  PedestrianManager.cpp
  Pedestrian.cpp
  PedestrianGeometry.cpp
  PedestrianFactory.cpp
  PorscheFFZ.cpp
  UDPBroadcast.cpp
  projects/radarcones.cpp
)


INCLUDE_DIRECTORIES(
   ../VehicleUtil
   ${OSGTERRAIN_INCLUDE_DIR}
   ${OSGCAL3D_INCLUDE_DIR}
   ${CAL3D_INCLUDE_DIR}
   ${GDAL_INCLUDE_DIR}
   ${V8_INCLUDE_DIR}
)


ADD_COVISE_LIBRARY(coTrafficSimulation SHARED ${SOURCES} ${HEADERS})
qt_use_modules(coTrafficSimulation Script ScriptTools)

COVISE_WNOERROR(coTrafficSimulation)

target_link_libraries(coTrafficSimulation
 ${OSGTERRAIN_LIBRARIES}
 ${OSGCAL_LIBRARIES}
 ${OPENSCENEGRAPH_LIBRARIES}
 ${EXTRA_LIBS}
 RoadTerrain
 #SteeringWheel #required for vrmlNodeVehicle (windows)
 coOpenVehicleUtil)
 

COVISE_INSTALL_TARGET(coTrafficSimulation)

IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(coTrafficSimulation "/Zc:wchar_t-")
ENDIF()


