USING(OpenXR)

set(HEADERS
  OpenXR.h
  DebugOutput.h
  GraphicsAPI.h
  GraphicsAPI_OpenGL.h
  HelperFunctions.h
  OpenXRDebugUtils.h
  OpenXRHelper.h
)
set(SOURCES
  OpenXR.cpp
  OpenXRDebugUtils.cpp
)

ADD_DEFINITIONS(-DXR_USE_GRAPHICS_API_OPENGL)

# build the plugin "OpenXR" from the ingredients listed in HEADERS and SOURCES
cover_add_plugin(OpenXR ${HEADERS} ${SOURCES})
# this should be handeled by USING OpenXR. this does not work with the current FindOpenXR: target_link_libraries(OpenXR OpenXR::openxr_loader)
