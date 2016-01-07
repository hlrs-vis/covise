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
   oscCancelCondition.h
   oscCancelConditionGroup.h
   oscCancelConditionRef.h
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
   oscConditionChoiceObject.h
   oscConditionChoiceRefObject.h
   oscConditionObject.h
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
   oscDriver.h
   oscDriverCatalog.h
   oscDriverRef.h
   oscEndCondition.h
   oscEndConditionGroup.h
   oscEndConditionRef.h
   oscEngine.h
   oscEntities.h
   oscEntity.h
   oscEntityAdd.h
   oscEntityCatalog.h
   oscEntityDelete.h
   oscEnvironment.h
   oscEnvironmentCatalog.h
   oscEnvironmentRef.h
   oscEvent.h
   oscEventStartCondition.h
   oscEventStartConditionGroup.h
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
   oscInitDynamics.h
   oscInitPosition.h
   oscInitState.h
   oscIntensity.h
   oscLaneChange.h
   oscLaneCoord.h
   oscLaneDynamics.h
   oscLaneOffset.h
   oscLaneOffsetDynamics.h
   oscLight.h
   oscLighting.h
   oscLights.h
   oscManeuver.h
   oscManeuverCatalog.h
   oscManeuverList.h
   oscManeuverRef.h
   oscManeuverRefActor.h
   oscMember.h
   oscMemberValue.h
   oscMirror.h
   oscMirrors.h
   oscMiscObject.h
   oscMiscObjectCatalog.h
   oscMiscObjectRef.h
   oscNamedObject.h
   oscNamedPriority.h
   oscNameRefId.h
   oscNotify.h
   oscNumericCondition.h
   oscObject.h
   oscObjectArrayVariable.h
   oscObjectBase.h
   oscObjectChoice.h  
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
   oscPerformance.h
   oscPosition.h
   oscPositionLane.h
   oscPositionRoad.h
   oscPositionRoute.h
   oscPositionWorld.h
   oscPositionXyz.h
   oscPrecipitation.h
   oscPriority.h
   oscReachPosition.h
   oscRefActor.h
   oscRefActorList.h
   oscReferenceHanding.h
   oscRelative.h
   oscRelativeChoice.h
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
   oscScenarioEnd.h
   oscSetController.h
   oscSetState.h
   oscShape.h
   oscSimulationTime.h
   oscSourceFile.h
   oscSpeed.h
   oscSpeedDynamics.h
   oscSpline.h
   oscStandsStill.h
   oscStartCondition.h
   oscStartConditionGroup.h
   oscStartConditionRef.h
   oscStoppingDistance.h
   oscStoryboard.h
   oscTest.h
   oscTime.h
   oscTimeHeadway.h
   oscTimeOfDay.h
   oscTimeToCollision.h
   oscTrafficJam.h
   oscTrafficLight.h
   oscTrafficSink.h
   oscTrafficSource.h
   oscUserData.h
   oscUserDefined.h
   oscUserDefinedAction.h
   oscUserDefinedCommand.h
   oscUserScript.h
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
   oscConditionChoiceObject.cpp
   oscConditionChoiceRefObject.cpp
   oscFactories.cpp
   oscMember.cpp
   oscMemberValue.cpp
   oscNamedPriority.cpp
   oscNumericCondition.cpp
   oscObjectBase.cpp
   oscPartner.cpp
   oscPedestrian.cpp
   oscPedestrianController.cpp
   oscPrecipitation.cpp
   oscPriority.cpp
   oscReachPosition.cpp
   oscReferenceHanding.cpp
   oscRoadCondition.cpp
   oscRoute.cpp
   oscShape.cpp
   oscSourceFile.cpp
   oscTrafficSource.cpp
   oscUserScript.cpp
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
