PROJECT(IvRenderer CXX C)

COVISE_FIND_PACKAGE(X11)
COVISE_FIND_PACKAGE(Motif)

Using(OpenGL)
Using(Virvo)
USING(INVENTOR MOTIF TIFF JPEG)

INCLUDE_DIRECTORIES(
  ../QtRenderer
  ${MOTIF_INCLUDE_DIR}
  ${X11_INCLUDE_DIR}
)

SET(SOURCES
  GLwDrawA.cpp 
  GLwMDrawA.cpp 
  ThumbWheel.cpp 
  logidrvr.cpp 
  BAWCursors.cpp 
  InvActiveNode.cpp 
  InvAnnotationEditor.cpp 
  InvAnnotationFlag.cpp 
  InvAnnotationManager.cpp 
  InvClipPlaneEditor.cpp 
  InvColEd.cpp 
  InvColormapManager.cpp 
  InvColPatch.cpp 
  InvColSlider.cpp 
  InvColWheel.cpp 
  InvCommunication.cpp 
  InvCoviseViewer.cpp 
  InvError.cpp 
  InvExaminerViewer.cpp 
  InvFloatCBList.cpp 
  InvFullViewer.cpp 
  InvMain.cpp 
  InvMtlPal.cpp 
  InvObjectList.cpp 
  InvObjectManager.cpp 
  InvOffscreenRenderer.cpp 
  InvPartEditor.cpp 
  InvPixmapButton.cpp 
  InvPlaneMover.cpp 
  InvPort.cpp 
  InvRenderer.cpp 
  InvRenderManager.cpp 
  InvSequencer.cpp 
  InvSimpleMtlEditor.cpp 
  InvSlider.cpp 
  InvTelePointer.cpp 
  InvTextureEd.cpp 
  InvThumbWhl.cpp 
  InvTimePart.cpp 
  InvTracker.cpp 
  InvUIRegion.cpp 
  InvViewer.cpp 
  ModuleInfo.cpp 
  polhemusdrvr.cpp 
  ShowL.cpp 
  SoXtLinuxMagellan.cpp 
  SoXtMagellan.cpp 
  TextField.cpp 
  coFeedbackParser.cpp
  InvTextManager.cpp
  ../QtRenderer/InvObjects.cpp
  ../QtRenderer/SoBillboard.cpp
  ../QtRenderer/InvManipList.cpp
)

if (COVISE_USE_VIRVO)
   set(SOURCES ${SOURCES}
      ../QtRenderer/SoVolume.cpp
      ../QtRenderer/SoVolumeDetail.cpp
  )
endif()

SET(EXTRASOURCES
  BAWCursors.h
  coFeedbackParser.h
  fastrak.h
  GLwDrawA.h
  GLwDrawAP.h
  GLwMDrawA.h
  GLwMDrawAP.h
  InvActiveNode.h
  InvAnnotationEditor.h
  InvAnnotationFlag.h
  InvAnnotationManager.h
  InvClipPlaneEditor.h
  InvColorEditor.h
  InvColormapManager.h
  InvColorPatch.h
  InvColorSlider.h
  InvColorWheel.h
  InvCommunication.h
  InvCoviseViewer.h
  InvDefs.h
  InvError.h
  InvExaminerViewer.h
  InvFloatCallbackList.h
  InvFullViewer.h
  InvMaterialPalette.h
  InvObjectEditor.h
  InvObjectList.h
  InvObjectManagerExtern.h
  InvObjectManager.h
  InvOffscreenRenderer.h
  InvPartEditor.h
  InvPixmapButton.h
  InvPlaneMover.h
  InvPort.h
  InvRenderer.h
  InvRenderManager.h
  InvSequencer.h
  InvSimpleMaterialEditor.h
  InvSlider.h
  InvTelePointer.h
  InvTextureEditor.h
  InvThumbWheel.h
  InvTimePart.h
  InvTimePartMultiHash.h
  InvTimer.h
  InvTracker.h
  InvUIRegion.h
  InvViewer.h
  InvTextManager.h
  logidrvr.h
  ModuleInfo.h
  polhemusdrvr.h
  ShowL.h
  SoXtLinuxMagellan.h
  SoXtMagellan.h
  TextField.h
  ThumbWheel.h
  ThumbWheelP.h
  xpm.h
)

ADD_DEFINITIONS("-DINVENTORRENDERER")

IF(APPLE)
  SET(EXTRA_LIBS ${EXTRA_LIBS} /opt/X11/lib/libGL.dylib)
ELSE(APPLE)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${OPENGL_LIBRARIES})
ENDIF(APPLE)

IF(CG_LIBRARY AND CG_GL_LIBRARY)
   SET(EXTRA_LIBS ${EXTRA_LIBS} ${CG_LIBRARY} ${CG_GL_LIBRARY})
ENDIF()

ADD_COVISE_MODULE(Renderer IvRenderer ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(IvRenderer
  coAppl coCore coUtil
  ${ZLIB_LIBRARY}
  ${EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT}
)

COVISE_WNOERROR(IvRenderer)
COVISE_INSTALL_TARGET(IvRenderer)
