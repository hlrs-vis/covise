find_package(Boost REQUIRED)
#find_package(GLUT_REQUIRED)

find_package(GLUT)
if(NOT GLUT_FOUND)
  message(STATUS "Note: not building vview (GLUT not found)")
  return()
endif()

deskvox_use_package(Boost)
deskvox_use_package(GLUT)

deskvox_link_libraries(virvo)
deskvox_link_libraries(virvo_fileio)

deskvox_add_tool(vview
  vvobjview.cpp
  vvobjview.h
  vvperformancetest.cpp
  vvperformancetest.h
  vvview.cpp
  vvview.h
)
