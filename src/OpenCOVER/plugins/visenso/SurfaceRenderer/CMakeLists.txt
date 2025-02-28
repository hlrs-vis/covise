IF(NOT PYTHON_FOUND)
  #MESSAGE("PYTHONLIBS not found! Skipping SurfaceRenderer")
  RETURN()
ENDIF()

SET(HEADERS
MobiusStrip.h
ParamSurface.h
SurfaceRenderer.h
SurfaceSphere.h
)

SET(SOURCES
MobiusStrip.cpp
ParamSurface.cpp
Sphere.cpp
SurfaceRenderer.cpp
SurfaceSphere.cpp
)
cover_add_plugin(SurfaceRenderer)
target_link_libraries(SurfaceRenderer Python::Python)
