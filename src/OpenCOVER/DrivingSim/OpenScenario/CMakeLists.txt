IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(PROJ4)
USING(XERCESC)

SET(HEADERS
   OpenScenarioBase.h
   oscAbsolute.h
   oscAcceleration.h
   oscAction.h
   oscAerodynamics.h
   oscArrayMember.h
   oscAutonomous.h
   oscAxles.h
   oscBehavior.h
   oscBody.h
   oscBoundingBox.h
   oscCatalogRef.h
   oscCatalogs.h
   oscCenter.h
   oscCharacterAppearance.h
   oscCharacterGesture.h
   oscCharacterMotion.h
   oscClothoid.h
   oscCog.h
   oscCollision.h
   oscColor.h
   oscCommand.h
   oscCondition.h
   oscContinuation.h
   oscControllerChoice.h
   oscCoord.h
   oscDate.h
   oscDimension.h
   oscDimensions.h
   oscDirectory.h
   oscDistance.h
   oscDistanceLateral.h
   oscDistanceLongitudinal.h
   oscDriverCatalog.h
   oscDriver.h
   oscDriverRef.h
   oscEngine.h
   oscEntity.h
   oscEntityCatalog.h
   oscEnvironmentCatalog.h
   oscEnvironment.h
   oscEvent.h
   oscExport.h
   oscEyepoint.h
   oscEyepoints.h
   oscFactories.h
   oscFactory.h
   oscFeature.h
   oscFeatures.h
   oscFile.h
   oscFilter.h
   oscFog.h
   oscFrustum.h
   oscGearbox.h
   oscGeneral.h
   oscHeader.h
   oscIntensity.h
   oscLaneChange.h
   oscLaneCoord.h
   oscLaneDynamics.h
   oscLaneOffset.h
   oscLaneOffsetDynamics.h
   oscLight.h
   oscLighting.h
   oscLights.h
   oscManeuverCatalog.h
   oscManeuver.h
   oscManeuverList.h
   oscMember.h
   oscMemberValue.h
   oscMirror.h
   oscMirrors.h
   oscMiscObjectCatalog.h
   oscMiscObject.h
   oscMiscObjectRef.h
   oscNamedObject.h
   oscNameRefId.h
   oscNumericCondition.h
   oscObject.h
   oscObjectBase.h
   oscObjectChoice.h
   oscObjectRef.h
   oscObjectVariable.h
   oscObserver.h
   oscObserverCatalog.h
   oscObserverId.h
   oscOffroad.h
   oscOrientation.h
   oscParameter.h
   oscParameterList.h
   oscParameters.h
   oscPartner.h
   oscPedestrian.h
   oscPedestrianCatalog.h
   oscPedestrianController.h
   oscPedestrianRef.h
   oscPedestrian.h
   oscPerformance.h
   oscPosition.h
   oscPositionLane.h
   oscPositionRoad.h
   oscPositionRoute.h
   oscPositionWorld.h
   oscPositionXyz.h
   oscPrecipitation.h
   oscReachPosition.h
   oscRefActor.h
   oscRelativeChoice.h
   oscReferenceHanding.h
   oscRelative.h
   oscRelativePositionLane.h
   oscRelativePositionRoad.h
   oscRelativePositionWorld.h
   oscRoadCondition.h
   oscRoadConditions.h
   oscRoadCoord.h
   oscRoadNetwork.h
   oscRoute.h
   oscRouting.h
   oscRoutingCatalog.h
   oscSetController.h
   oscSetState.h
   oscSimulationTime.h
   oscShape.h
   oscSpeedDynamics.h
   oscSpeed.h
   oscSpline.h
   oscStandsStill.h
   oscStartConditionGroup.h
   oscStartCondition.h
   oscStoppingDistance.h
   oscTest.h
   oscTime.h
   oscTimeHeadway.h
   oscTimeOfDay.h
   oscTimeToCollision.h
   oscTrafficLight.h
   oscUserData.h
   oscUserDefined.h
   oscVariables.h
   oscVehicle.h
   oscVehicleAxle.h
   oscVehicleCatalog.h
   oscVehicleRef.h
   oscVelocity.h
   oscVisibility.h
   oscWaypoint.h
   oscWeather.h
)

SET(SOURCES
   OpenScenarioBase.cpp
   oscArrayMember.cpp
   oscBody.cpp
   oscEvent.cpp
   oscFactories.cpp
   oscMember.cpp
   oscMemberValue.cpp
   oscNumericCondition.cpp
   oscObject.cpp
   oscObjectBase.cpp
   oscObjectRef.cpp
   oscPartner.cpp
   oscPedestrian.cpp
   oscPedestrianController.cpp
   oscPrecipitation.cpp
   oscReachPosition.cpp
   oscReferenceHanding.cpp
   oscRoadCondition.cpp
   oscRoute.cpp
   oscShape.cpp
   oscVariables.cpp
   oscVehicle.cpp
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
