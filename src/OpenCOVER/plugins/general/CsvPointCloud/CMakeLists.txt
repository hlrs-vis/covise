using(VRML)
using(boost)
covise_find_package(Boost COMPONENTS timer REQUIRED)

SET(HEADERS
  DataTable.h  
  exprtk.hpp
  CsvPointCloud.h
)

SET(SOURCES
  DataTable.cpp  
  CsvPointCloud.cpp
)

cover_add_plugin(CsvPointCloud)
target_link_libraries(CsvPointCloud Boost::timer)
if(MSVC)
  target_compile_options(CsvPointCloud PRIVATE /bigobj)
endif()
