SET(HEADERS
  CuttingSurfacePlugin.h
  CuttingSurfaceInteraction.h
  CuttingSurfacePlane.h
  CuttingSurfaceCylinder.h
  CuttingSurfaceSphere.h
  CuttingSurfaceBox.h
)
SET(SOURCES
  CuttingSurfacePlugin.cpp
  CuttingSurfaceInteraction.cpp
  CuttingSurfacePlane.cpp
  CuttingSurfaceCylinder.cpp
  CuttingSurfaceSphere.cpp
  CuttingSurfaceBox.cpp
)
ADD_DEFINITIONS("-DNOCP3DTEX")
cover_add_plugin(CuttingSurfacePlugin)

SET_TARGET_PROPERTIES(CuttingSurfacePlugin PROPERTIES OUTPUT_NAME CuttingSurface)
