IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(PROJ4)
USING(XERCESC)

SET(HEADERS
   OpenScenarioBase.h
   oscAbsolute.h
   oscAction.h
   oscArrayMember.h
   oscAutonomous.h
   oscBehavior.h
   oscBody.h
   oscBoundingBox.h
   oscCatalog.h
   oscCatalogRef.h
   oscCatalogs.h
   oscCenter.h
   oscColor.h
   oscCondition.h
   oscCoord.h
   oscDate.h
   oscDimension.h
   oscDimensions.h
   oscDirectory.h
   oscDriverCatalog.h
   oscDriver.h
   oscEntityCatalog.h
   oscEnvironmentCatalog.h
   oscEnvironment.h
   oscEvent.h
   oscExport.h
   oscFactories.h
   oscFactory.h
   oscFile.h
   oscFilter.h
   oscFog.h
   oscFrustum.h
   oscHeader.h
   oscIntensity.h
   oscLaneChange.h
   oscLaneCoord.h
   oscLaneDynamics.h
   oscLight.h
   oscManeuverCatalog.h
   oscManeuver.h
   oscManeuverList.h
   oscMember.h
   oscMemberValue.h
   oscMiscObjectCatalog.h
   oscMiscObject.h
   oscNamedObject.h
   oscNameId.h
   oscObjectBase.h
   oscObjectVariable.h
   oscObserver.h
   oscObserverCatalog.h
   oscOrientation.h
   oscParameter.h
   oscParameterList.h
   oscParameters.h
   oscPedestrianCatalog.h
   oscPedestrian.h
   oscPosition.h
   oscPositionLane.h
   oscPositionRoad.h
   oscPositionRoute.h
   oscPositionWorld.h
   oscPositionXyz.h
   oscPrecipitation.h
   oscRefActor.h
   oscRelativeChoice.h
   oscRelative.h
   oscRelativePositionLane.h
   oscRelativePositionRoad.h
   oscRelativePositionWorld.h
   oscRoadCondition.h
   oscRoadConditions.h
   oscRoadCoord.h
   oscRoadNetwork.h
   oscRoutingCatalog.h
   oscSpeedDynamics.h
   oscSpeed.h
   oscStartConditionGroup.h
   oscStartCondition.h
   oscTime.h
   oscTimeOfDay.h
   oscUserData.h
   oscVariables.h
   oscVehicleAxle.h
   oscVehicleCatalog.h
   oscWeather.h
)

SET(SOURCES
   OpenScenarioBase.cpp
   oscArrayMember.cpp
   oscBody.cpp
   oscEvent.cpp
   oscFactories.cpp
   oscHeader.cpp
   oscMember.cpp
   oscMemberValue.cpp
   oscNamedObject.cpp
   oscObjectBase.cpp
   oscPedestrian.cpp
   oscPrecipitation.cpp
   oscRoadCondition.cpp
   oscVariables.cpp
   oscWeather.cpp
)

IF(WIN32)
  set(EXTRA_LIBS ${EXTRA_LIBS} ws2_32.lib)
ENDIF(WIN32)

 


   INCLUDE_DIRECTORIES(
      ./..
      ${PROJ4_INCLUDE_DIR}
      ${V8_INCLUDE_DIR}
   )

   ADD_COVISE_LIBRARY(coOpenScenario SHARED ${SOURCES} ${HEADERS})

   COVISE_WNOERROR(coOpenScenario)

   COVISE_INSTALL_TARGET(coOpenScenario)
target_link_libraries(coOpenScenario ${EXTRA_LIBS})

IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(coOpenScenario "/Zc:wchar_t-")
ENDIF()



# test application

SET(TESTSOURCES
   testOpenScenario.cpp
)

add_covise_executable(oscTest MACOSX_BUNDLE ${TESTSOURCES} ${TESTHEADERS})

covise_install_target(oscTest)

target_link_libraries(oscTest coOpenScenario ${EXTRA_LIBS})
