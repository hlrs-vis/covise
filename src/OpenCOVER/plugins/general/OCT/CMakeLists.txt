#OpenCOVER plugins calit2 PointCloud

find_file(EXPRTK_FOUND exprtk.hpp ${EXTERNLIBS} ${EXTERNLIBS}/exprtk ${EXTERNLIBS}/ALL)
if(NOT EXPRTK_FOUND)
  return()
endif()
SET(HEADERS
  DataTable.h  
  Oct.h
)

SET(SOURCES
  DataTable.cpp  
  Oct.cpp
)

cover_add_plugin(Oct)
get_filename_component(EXPRTK_DIR ${EXPRTK_FOUND} DIRECTORY)
target_include_directories(Oct PRIVATE ${EXPRTK_DIR})
if(MSVC)
  target_compile_options(Oct PRIVATE /bigobj)
endif()
