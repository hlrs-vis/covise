using(VRML)
using(boost)
covise_find_package(Boost COMPONENTS timer REQUIRED)

SET(HEADERS
  ColorMapShader.h
  CsvPointCloud.h
  DataTable.h  
  exprtk.hpp
  CsvPointCloud.h
)

SET(SOURCES
  ColorMapShader.cpp
  CsvPointCloud.cpp
  DataTable.cpp
)

cover_add_plugin(CsvPointCloud)
target_link_libraries(CsvPointCloud Boost::timer)
if(MSVC)
  target_compile_options(CsvPointCloud PRIVATE /bigobj)
endif()
