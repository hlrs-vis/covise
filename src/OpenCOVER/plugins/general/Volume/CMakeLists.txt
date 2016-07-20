USING(Virvo)
USING(Virvo:osg)

SET(HEADERS
  VolumePlugin.h
  coAlphaHatPin.h
  coAlphaBlankPin.h
  coClipSphere.h
  coDefaultFunctionEditor.h
  coFunctionEditor.h
  coHSVPin.h
  coHSVSelector.h
  coPin.h
  coPinEditor.h
  coPresetButtonGeometry.h
  coPreviewCube.h
)
SET(SOURCES
  VolumePlugin.cpp 
  coAlphaHatPin.cpp 
  coAlphaBlankPin.cpp 
  coClipSphere.cpp
  coDefaultFunctionEditor.cpp 
  coFunctionEditor.cpp 
  coHSVPin.cpp 
  coHSVSelector.cpp 
  coPin.cpp 
  coPinEditor.cpp 
  coPresetButtonGeometry.cpp 
  coPreviewCube.cpp 
)

INCLUDE_DIRECTORIES(
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
  )

IF(CUDA_FOUND)
  ADD_DEFINITIONS("-DHAVE_CUDA -DNV_PROPRIETARY_CODE")
ENDIF(CUDA_FOUND)

cover_add_plugin(Volume)
