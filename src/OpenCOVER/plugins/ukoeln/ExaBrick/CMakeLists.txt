USING(ExaBrick)
SET(HEADERS
  ExaBrickPlugin.h
  # coDefaultFunctionEditor.h
  # coFunctionEditor.h
  #ExaBrickGlobals.h
  # mat4.h
)
SET(SOURCES
  ExaBrickPlugin.cpp
  # coDefaultFunctionEditor.cpp
  # coFunctionEditor.cpp
  # Renderer.cpp
  # mat4.cpp
)
# INCLUDE_DIRECTORIES(
#     ${OPENGL_INCLUDE_DIR}
#     ${GLEW_INCLUDE_DIR}
#   )
cover_add_plugin(ExaBrickPlugin ${HEADERS} ${SOURCES})
COVISE_WNOERROR(ExaBrickPlugin)
