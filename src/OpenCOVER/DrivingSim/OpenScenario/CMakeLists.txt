IF(NOT COVISE_BUILD_DRIVINGSIM)
  RETURN()
ENDIF()

USING(PROJ4)
USING(XERCESC)
USING(BOOST)

SET(HEADERS
   OpenScenarioBase.h
   oscAbsoluteTypeA.h
   oscAbsoluteTypeB.h
   oscAction.h
   oscActions.h
   oscAdditionalAxles.h
   oscAerodynamics.h
   oscAfterManeuvers.h
   oscAutonomous.h
   oscAxles.h
   oscBehavior.h
   oscBody.h
   oscBoundingBox.h
   oscCancelConditionsGroupsTypeA.h
   oscCancelConditionsGroupTypeA.h
   oscCancelConditionsTypeA.h
   oscCancelConditionsTypeB.h
   oscCatalogBase.h
   oscCatalogObject.h
   oscCatalogReferenceTypeA.h
   oscCatalogs.h
   oscCenter.h
   oscCharacterAppearance.h
   oscCharacterGesture.h
   oscCharacterMotion.h
   oscChoiceController.h
   oscChoiceObject.h
   oscClothoid.h
   oscCog.h
   oscCollision.h
   oscColor.h
   oscCommand.h
   oscConditionBase.h
   oscConditionChoiceBase.h
   oscConditionChoiceTypeA.h
   oscConditionTypeA.h
   oscConditionTypeB.h
   oscContinuation.h
   oscControlPoint.h
   oscCoord.h
   oscCurrentPosition.h
   oscDate.h
   oscDimensionTypeA.h
   oscDimensionTypeB.h
   oscDirectory.h
   oscDistance.h
   oscDistanceLateral.h
   oscDistanceLongitudinal.h
   oscDriver.h
   oscEndConditionsGroupsTypeA.h
   oscEndConditionsGroupTypeA.h
   oscEndConditionsTypeA.h
   oscEndConditionsTypeB.h
   oscEndOfRoad.h
   oscEngine.h
   oscEntities.h
   oscEntity.h
   oscEntityAdd.h
   oscEntityDelete.h
   oscEnvironment.h
   oscEnvironmentReference.h
   oscEvent.h
   oscEvents.h
   oscExport.h
   oscEyepoint.h
   oscEyepoints.h
   oscFactories.h
   oscFactory.h
   oscFeature.h
   oscFeatures.h
   oscFile.h
   oscFileHeader.h
   oscFilter.h
   oscFilters.h
   oscFog.h
   oscFollowRoute.h
   oscFrustum.h
   oscGearbox.h
   oscGeneral.h
   oscInitDynamics.h
   oscInitState.h
   oscIntensity.h
   oscLaneChange.h
   oscLaneCoord.h
   oscLaneDynamics.h
   oscLaneOffset.h
   oscLaneOffsetDynamics.h
   oscLateral.h
   oscLight.h
   oscLighting.h
   oscLights.h
   oscLongitudinal.h
   oscManeuverGroupsTypeAB.h
   oscManeuverGroupTypeAB.h
   oscManeuversTypeB.h
   oscManeuversTypeC.h
   oscManeuverTypeA.h
   oscManeuverTypeB.h
   oscManeuverTypeC.h
   oscMember.h
   oscMemberArray.h
   oscMemberCatalog.h
   oscMemberValue.h
   oscMirror.h
   oscMirrors.h
   oscMiscObject.h
   oscNamePriority.h
   oscNameRefId.h
   oscNameRefIdUserData.h
   oscNameUserData.h
   oscNone.h
   oscNotify.h
   oscNumericCondition.h
   oscObject.h
   oscObjectBase.h
   oscObjects.h
   oscObjectVariable.h
   oscObjectVariableArray.h
   oscObserverId.h
   oscObserverTypeA.h
   oscObserverTypeB.h
   oscOffroad.h
   oscOrientation.h
   oscOrientationOptional.h
   oscParameterListTypeA.h
   oscParameterListTypeB.h
   oscParameterTypeA.h
   oscParameterTypeB.h
   oscPartnerObject.h
   oscPartnerType.h
   oscPedestrian.h
   oscPerformance.h
   oscPolyline.h
   oscPosition.h
   oscPositionLane.h
   oscPositionRoad.h
   oscPositionRoute.h
   oscPositionWorld.h
   oscPositionXyz.h
   oscPrecipitation.h
   oscPriority.h
   oscReachPosition.h
   oscRefActorsTypeA.h
   oscRefActorsTypeB.h
   oscRefActorTypeA.h
   oscRefActorTypeB.h
   oscReferenceHandling.h
   oscRelativePositionLane.h
   oscRelativePositionRoad.h
   oscRelativePositionWorld.h
   oscRelativeTypeA.h
   oscRelativeTypeB.h
   oscRelativeTypeC.h
   oscRoadCondition.h
   oscRoadConditions.h
   oscRoadConditionsGroup.h
   oscRoadCoord.h
   oscRoadNetwork.h
   oscRoute.h
   oscRouting.h
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
   oscStartConditionsGroupsTypeA.h
   oscStartConditionsGroupsTypeC.h
   oscStartConditionsGroupTypeA.h
   oscStartConditionsGroupTypeC.h
   oscStartConditionsTypeA.h
   oscStartConditionsTypeB.h
   oscStartConditionsTypeC.h
   oscStartConditionTypeC.h
   oscStoppingDistance.h
   oscStoryboard.h
   oscTest.h
   oscTime.h
   oscTimeHeadway.h
   oscTimeOfDay.h
   oscTimeToCollision.h
   oscTiming.h
   oscTrafficJam.h
   oscTrafficLight.h
   oscTrafficSink.h
   oscTrafficSource.h
   oscUserData.h
   oscUserDataList.h
   oscUserDefined.h
   oscUserDefinedAction.h
   oscUserDefinedCommand.h
   oscUserScript.h
   oscUtilities.h
   oscVariables.h
   oscVehicle.h
   oscVehicleAxle.h
   oscVisibility.h
   oscVisualRange.h
   oscWaypoint.h
   oscWaypoints.h
   oscWeather.h
)

SET(SOURCES
   OpenScenarioBase.cpp
   oscBody.cpp
   oscConditionChoiceBase.cpp
   oscConditionChoiceTypeA.cpp
   oscControlPoint.cpp
   oscFactories.cpp
   oscLateral.cpp
   oscMember.cpp
   oscMemberArray.cpp
   oscMemberCatalog.cpp
   oscMemberValue.cpp
   oscNamePriority.cpp
   oscNumericCondition.cpp
   oscObjectBase.cpp
   oscPartnerType.cpp
   oscPedestrian.cpp
   oscPedestrianController.cpp
   oscPrecipitation.cpp
   oscPriority.cpp
   oscReachPosition.cpp
   oscReferenceHandling.cpp
   oscRoadCondition.cpp
   oscRoute.cpp
   oscShape.cpp
   oscSourceFile.cpp
   oscTiming.cpp
   oscTrafficSource.cpp
   oscUserScript.cpp
   oscUtilities.cpp
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
