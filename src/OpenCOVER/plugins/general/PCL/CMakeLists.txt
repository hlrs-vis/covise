#Point cloud rendering using PCL

USING(PCL)
USING(EIGEN)
USING(Boost)

SET(HEADERS
  PCLPlugin.h
  common.h
  outofcore.h
  outofcore_impl.h
  outofcore_octree_reader.h
  point_cloud.h
  shapes.h
  surfel.h
  utility_point_types.h
  impl/surfel.hpp
  impl/point_cloud.hpp
)

SET(SOURCES
  PCLPlugin.cpp
  point_cloud.cpp
  shapes.cpp
  surfel.cpp
  common.cpp
)

cover_add_plugin(PCL)
