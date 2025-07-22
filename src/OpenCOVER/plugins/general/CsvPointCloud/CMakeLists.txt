using(VRML)
using(boost)
covise_find_package(Boost COMPONENTS timer REQUIRED)

SET(HEADERS
  ColorMapShader.h
  CsvPointCloud.h
  DataTable.h  
  OctUtils.h
  RenderObject.h
  SurfacePrimitiveSet.h
)

SET(SOURCES
  ColorMapShader.cpp
  CsvPointCloud.cpp
  DataTable.cpp
  OctUtils.cpp
  RenderObject.cpp  
)

cover_add_plugin(CsvPointCloud)
target_link_libraries(CsvPointCloud Boost::timer exprtk)
if(MSVC)
  target_compile_options(CsvPointCloud PRIVATE /bigobj)
endif()
