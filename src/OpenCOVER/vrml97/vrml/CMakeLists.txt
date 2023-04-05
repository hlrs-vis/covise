# @file
# 
# CMakeLists.txt for kernel - vrml97 - vrml
#
# @author Blasius Czink

ADD_DEFINITIONS(-DVRML_EXPORT)

ADD_DEFINITIONS(-DHAVE_CONFIG_H -DYYERROR_VERBOSE)
# TODO: the following defines might be necessary on some platforms
# ADD_DEFINITIONS(-D_GNU_SOURCE -DYYINCLUDED_STDLIB_H)
# handle them with cmake feature tests or simply with
# IF(${CMAKE_SYSTEM_NAME} MATCHES "whater_system_name")

# TODO: create Find-Modules for esd, libwww, arts, oss, irixal

# gather optional includes in EXTRA_INCS
SET(EXTRA_INCS )
# gather optional libraries in EXTRA_LIBS
SET(EXTRA_LIBS )
using(JPEG)
using(PNG)
using(TIFF)
using(ZLIB)

IF(UNIX AND NOT APPLE)
  FIND_PACKAGE(ALSA QUIET)
  IF(ALSA_FOUND)
    ADD_DEFINITIONS(-DHAVE_ALSA)
    SET(EXTRA_INCS ${EXTRA_INCS} ${ALSA_INCLUDE_DIR})
    SET(EXTRA_LIBS ${EXTRA_LIBS} ${ALSA_LIBRARIES})
  ENDIF()
ENDIF()

FIND_PACKAGE(OpenAL QUIET)
IF(OPENAL_FOUND)
  ADD_DEFINITIONS(-DHAVE_OPENAL)
  SET(EXTRA_INCS ${EXTRA_INCS} ${OPENAL_INCLUDE_DIR})
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${OPENAL_LIBRARY})

  # we can also make use of ALUT
  FIND_PACKAGE(ALUT)
  IF(ALUT_FOUND)
    ADD_DEFINITIONS(-DHAVE_ALUT)
    SET(EXTRA_INCS ${EXTRA_INCS} ${ALUT_INCLUDE_DIR})
    SET(EXTRA_LIBS ${EXTRA_LIBS} ${ALUT_LIBRARIES})
  ENDIF()
ENDIF()

FIND_PACKAGE(AUDIOFILE QUIET)
IF(AUDIOFILE_FOUND)
  ADD_DEFINITIONS(-DHAVE_AUDIOFILE)
  SET(EXTRA_INCS ${EXTRA_INCS} ${AUDIOFILE_INCLUDE_DIR})
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${AUDIOFILE_LIBRARIES})
ENDIF()

IF(CURL_FOUND)
  SET(EXTRA_INCS ${EXTRA_INCS} ${CURL_INCLUDE_DIR})
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${CURL_LIBRARIES})
ENDIF()

IF(PTHREAD_FOUND)
    ADD_DEFINITIONS(-DHAVE_PTHREAD)
ENDIF()

ADD_DEFINITIONS(-DYY_NEVER_INTERACTIVE)
using(FLEX)
covise_find_package(BISON)
IF(UNIX)
  ADD_DEFINITIONS(-DXP_UNIX)
endif()
BISON_TARGET(VrmlParser parser.yy ${CMAKE_CURRENT_BINARY_DIR}/parser_yacc.cpp COMPILE_FLAGS "--defines=${CMAKE_CURRENT_BINARY_DIR}/parser_yacc.hpp -y -t -p parser -b parser")
FLEX_TARGET(VrmlScanner lexer.ll  ${CMAKE_CURRENT_BINARY_DIR}/lexer_lex.cpp COMPILE_FLAGS "-L -s -Plexer")
ADD_FLEX_BISON_DEPENDENCY(VrmlScanner VrmlParser)
# --nounput not available in older versions
#  FLEX_TARGET(VrmlScanner lexer.ll  ${CMAKE_CURRENT_BINARY_DIR}/lexer_lex.cpp COMPILE_FLAGS "-L -s --nounput -Plexer")
  
# on windows we just use existing files
#SET(BISON_VrmlParser_OUTPUTS win32/parser_yacc.cpp)
#SET(FLEX_VrmlScanner_OUTPUTS win32/lexer_lex.cpp)
#INCLUDE_DIRECTORIES( win32) 

INCLUDE_DIRECTORIES(
  ${EXTRA_INCS}
  "${CMAKE_CURRENT_BINARY_DIR}/../js"
)

SET(VRML_SOURCES
  gifread.c
  jpgread.c
  pngread.c
  tifread.c
  coEventQueue.cpp
  System.cpp
  Audio.cpp
  Doc.cpp
  Image.cpp
  MathUtils.cpp
  mpgread.cpp
  ScriptJDK.cpp
  ScriptJS.cpp
  ScriptObject.cpp
  Viewer.cpp
  VrmlField.cpp
  VrmlNamespace.cpp
  VrmlNodeAnchor.cpp
  VrmlNodeAppearance.cpp
  VrmlNodeAudioClip.cpp
  VrmlNodeBackground.cpp
  VrmlNodeBillboard.cpp
  VrmlNodeBooleanSequencer.cpp
  VrmlNodeBox.cpp
  VrmlNodeBumpMapping.cpp
  VrmlNodeCollision.cpp
  VrmlNodeColor.cpp
  VrmlNodeColorRGBA.cpp
  VrmlNodeColorInt.cpp
  VrmlNodeColoredSet.cpp
  VrmlNodeCone.cpp
  VrmlNodeCoordinate.cpp
  VrmlNodeCoordinateInt.cpp
  VrmlNodeCOVER.cpp
  VrmlNode.cpp
  VrmlNodeCubeTexture.cpp
  VrmlNodeCylinder.cpp
  VrmlNodeCylinderSensor.cpp
  VrmlNodeDirLight.cpp
  VrmlNodeElevationGrid.cpp
  VrmlNodeExtrusion.cpp
  VrmlNodeFog.cpp
  VrmlNodeFontStyle.cpp
  VrmlNodeGeometry.cpp
  VrmlNodeGroup.cpp
  VrmlNodeIFaceSet.cpp
  VrmlNodeILineSet.cpp
  VrmlNodeIPolygonsCommon.cpp
  VrmlNodeIQuadSet.cpp
  VrmlNodeITriangleFanSet.cpp
  VrmlNodeITriangleSet.cpp
  VrmlNodeITriangleStripSet.cpp
  VrmlNodeImageTexture.cpp
  VrmlNodeIndexedSet.cpp
  VrmlNodeInline.cpp
  VrmlNodeLight.cpp
  VrmlNodeLOD.cpp
  VrmlNodeMaterial.cpp
  VrmlNodeMetadata.cpp
  VrmlNodeMetadataBoolean.cpp
  VrmlNodeMetadataDouble.cpp
  VrmlNodeMetadataFloat.cpp
  VrmlNodeMetadataInteger.cpp
  VrmlNodeMetadataString.cpp
  VrmlNodeMetadataSet.cpp
  VrmlNodeMovieTexture.cpp
  VrmlNodeMultiTexture.cpp
  VrmlNodeMultiTextureCoordinate.cpp
  VrmlNodeMultiTextureTransform.cpp
  VrmlNodeNavigationInfo.cpp
  VrmlNodeNormal.cpp
  VrmlNodeNormalInt.cpp
  VrmlNodeOrientationInt.cpp
  VrmlNodePixelTexture.cpp
  VrmlNodePlaneSensor.cpp
  VrmlNodePointLight.cpp
  VrmlNodePointSet.cpp
  VrmlNodePolygonsCommon.cpp
  VrmlNodePositionInt.cpp
  VrmlNodeProto.cpp
  VrmlNodeProximitySensor.cpp
  VrmlNodeQuadSet.cpp
  VrmlNodeScalarInt.cpp
  VrmlNodeScript.cpp
  VrmlNodeShape.cpp
  VrmlNodeSound.cpp
  VrmlNodeSpaceSensor.cpp
  VrmlNodeSphere.cpp
  VrmlNodeSphereSensor.cpp
  VrmlNodeSpotLight.cpp
  VrmlNodeSwitch.cpp
  VrmlNodeText.cpp
  VrmlNodeTexture.cpp
  VrmlNodeTextureCoordinate.cpp
  VrmlNodeTextureCoordinateGenerator.cpp
  VrmlNodeTextureTransform.cpp
  VrmlNodeTimeSensor.cpp
  VrmlNodeTouchSensor.cpp
  VrmlNodeTransform.cpp
  VrmlNodeTriangleFanSet.cpp
  VrmlNodeTriangleSet.cpp
  VrmlNodeTriangleStripSet.cpp
  VrmlNodeType.cpp
  VrmlNodeViewpoint.cpp
  VrmlNodeVisibilitySensor.cpp
  VrmlNodeWave.cpp
  VrmlNodeWorldInfo.cpp
  VrmlScene.cpp
  PlayerAlsa.cpp
  PlayerArts.cpp
  PlayerAServer.cpp
  PlayerFactory.cpp
  Player.cpp
  PlayerEsd.cpp
  PlayerIrixAL.cpp
  PlayerMix.cpp
  PlayerOpenAL.cpp
  PlayerOSS.cpp
)

SET(VRML_HEADERS
  Audio.h
  Byteswap.h
  coEventQueue.h
  config.h
  Doc.h
  gifread.h
  Image.h
  jpgread.h
  Listener.h
  MathUtils.h
  mpgread.h
  Player.h
  PlayerAlsa.h
  PlayerArts.h
  PlayerAServer.h
  PlayerEsd.h
  PlayerFactory.h
  PlayerIrixAL.h
  PlayerMix.h
  PlayerOpenAL.h
  PlayerOSS.h
  pngread.h
  ScriptJDK.h
  ScriptJS.h
  ScriptObject.h
  System.h
  tifread.h
  Viewer.h
  VrmlEvent.h
  vrmlexport.h
  VrmlField.h
  VrmlLinMath.h
  VrmlMFBool.h
  VrmlMFColor.h
  VrmlMFColorRGBA.h
  VrmlMFDouble.h
  VrmlMFFloat.h
  VrmlMFInt.h
  VrmlMFNode.h
  VrmlMFRotation.h
  VrmlMFString.h
  VrmlMFTime.h
  VrmlMFVec2d.h
  VrmlMFVec2f.h
  VrmlMFVec3d.h
  VrmlMFVec3f.h
  VrmlNamespace.h
  VrmlNodeBindable.h
  VrmlNodeChild.h
  VrmlNodeTexture.h
  VrmlNode.h
  VrmlNodeAnchor.h
  VrmlNodeAppearance.h
  VrmlNodeAudioClip.h
  VrmlNodeBackground.h
  VrmlNodeBillboard.h
  VrmlNodeBooleanSequencer.h
  VrmlNodeBox.h
  VrmlNodeBumpMapping.h
  VrmlNodeCollision.h
  VrmlNodeColor.h
  VrmlNodeColorRGBA.h
  VrmlNodeColorInt.h
  VrmlNodeColoredSet.h
  VrmlNodeCone.h
  VrmlNodeCoordinate.h
  VrmlNodeCoordinateInt.h
  VrmlNodeCOVER.h
  VrmlNode.h
  VrmlNodeCubeTexture.h
  VrmlNodeCylinder.h
  VrmlNodeCylinderSensor.h
  VrmlNodeDirLight.h
  VrmlNodeElevationGrid.h
  VrmlNodeExtrusion.h
  VrmlNodeFog.h
  VrmlNodeFontStyle.h
  VrmlNodeGeometry.h
  VrmlNodeGroup.h
  VrmlNodeIFaceSet.h
  VrmlNodeILineSet.h
  VrmlNodeIPolygonsCommon.h
  VrmlNodeIQuadSet.h
  VrmlNodeITriangleFanSet.h
  VrmlNodeITriangleSet.h
  VrmlNodeITriangleStripSet.h
  VrmlNodeImageTexture.h
  VrmlNodeIndexedSet.h
  VrmlNodeInline.h
  VrmlNodeLight.h
  VrmlNodeLOD.h
  VrmlNodeMaterial.h
  VrmlNodeMetadata.h
  VrmlNodeMetadataDouble.h
  VrmlNodeMetadataFloat.h
  VrmlNodeMetadataInteger.h
  VrmlNodeMetadataString.h
  VrmlNodeMetadataSet.h
  VrmlNodeMovieTexture.h
  VrmlNodeMultiTexture.h
  VrmlNodeMultiTextureCoordinate.h
  VrmlNodeMultiTextureTransform.h
  VrmlNodeNavigationInfo.h
  VrmlNodeNormal.h
  VrmlNodeNormalInt.h
  VrmlNodeOrientationInt.h
  VrmlNodePixelTexture.h
  VrmlNodePlaneSensor.h
  VrmlNodePointLight.h
  VrmlNodePointSet.h
  VrmlNodePolygonsCommon.h
  VrmlNodePositionInt.h
  VrmlNodeProto.h
  VrmlNodeProximitySensor.h
  VrmlNodeQuadSet.h
  VrmlNodeScalarInt.h
  VrmlNodeScript.h
  VrmlNodeShape.h
  VrmlNodeSound.h
  VrmlNodeSpaceSensor.h
  VrmlNodeSphere.h
  VrmlNodeSphereSensor.h
  VrmlNodeSpotLight.h
  VrmlNodeSwitch.h
  VrmlNodeText.h
  VrmlNodeTextureCoordinate.h
  VrmlNodeTextureCoordinateGenerator.h
  VrmlNodeTextureTransform.h
  VrmlNodeTimeSensor.h
  VrmlNodeTouchSensor.h
  VrmlNodeTransform.h
  VrmlNodeTriangleFanSet.h
  VrmlNodeTriangleSet.h
  VrmlNodeTriangleStripSet.h
  VrmlNodeType.h
  VrmlNodeViewpoint.h
  VrmlNodeVisibilitySensor.h
  VrmlNodeWave.h
  VrmlNodeWorldInfo.h
  VrmlScene.h
  VrmlSFBool.h
  VrmlSFColor.h
  VrmlSFColorRGBA.h
  VrmlSFDouble.h
  VrmlSFFloat.h
  VrmlSFImage.h
  VrmlSFInt.h
  VrmlSFNode.h
  VrmlSFRotation.h
  VrmlSFString.h
  VrmlSFTime.h
  VrmlSFVec2d.h
  VrmlSFVec2f.h
  VrmlSFVec3d.h
  VrmlSFVec3f.h
  VrmlSFMatrix.h
)


SET(EXTRASOURCES
  parser.yy
  lexer.ll
)

IF(UNIX)
ADD_COVISE_LIBRARY(coVRML SHARED ${VRML_SOURCES} ${VRML_HEADERS} ${BISON_VrmlParser_OUTPUTS} ${FLEX_VrmlScanner_OUTPUTS})
ELSE()
ADD_COVISE_LIBRARY(coVRML SHARED ${EXTRASOURCES} ${VRML_SOURCES} ${VRML_HEADERS} ${BISON_VrmlParser_OUTPUTS} ${FLEX_VrmlScanner_OUTPUTS})
set(EXTRA_LIBS ${EXTRA_LIBS} ws2_32.lib)
ENDIF()
#IF(UNIX)
#  ADD_COVISE_COMPILE_FLAGS(coVRML "-fPIC -fno-strict-aliasing")
#ENDIF()

if(APPLE)
   add_covise_compile_flags(coVRML "-Wno-deprecated")
   add_covise_compile_flags(coVRML "-Wno-error=shift-negative-value")
   add_covise_compile_flags(coVRML "-Wno-error=unknown-warning-option")
endif()

if (NOT MSVC)
    set_source_files_properties(${FLEX_VrmlScanner_OUTPUTS} PROPERTIES COMPILE_FLAGS -Wno-error)
    set_source_files_properties(${BISON_VrmlParser_OUTPUTS} PROPERTIES COMPILE_FLAGS -Wno-error)
endif()

COVISE_WNOERROR(coVRML BASEARCH chuckwalla stromboli waran gecko skink zebu dornteufel lurchi basilisk neuffen leopard)

TARGET_LINK_LIBRARIES(coVRML
  coJS ${COVISE_UTIL_LIBRARY}
  ${EXTRA_LIBS}
  ${CMAKE_THREAD_LIBS_INIT}
)

COVISE_INSTALL_TARGET(coVRML)
