
SET(HEADERS
  DataTable.h  
  exprtk.hpp
  CsvPointCloud.h
)

SET(SOURCES
  DataTable.cpp  
  CsvPointCloud.cpp
)

cover_add_plugin(Oct)
if(MSVC)
  target_compile_options(CsvPointCloud PRIVATE /bigobj)
endif()
