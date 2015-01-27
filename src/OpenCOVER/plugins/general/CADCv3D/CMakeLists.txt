#OpenCOVER plugins general CADCv3D

SET(HEADERS
  CADCv3DColor.h
  CADCv3DGeoList.h
  CADCv3DGeometry.h
  CADCv3DPlugin.h
  CADCv3DPrimitive.h
  CADCv3DSerializable.h
  CADCv3DVertex.h
)

SET(SOURCES
  CADCv3DColor.cpp
  CADCv3DGeoList.cpp
  CADCv3DGeometry.cpp
  CADCv3DPlugin.cpp
  CADCv3DPrimitive.cpp
  CADCv3DVertex.cpp
)

cover_add_plugin(CADCv3D)
