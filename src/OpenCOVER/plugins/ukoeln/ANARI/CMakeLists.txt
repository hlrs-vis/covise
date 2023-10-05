covise_find_package(anari)
if(NOT ANARI_FOUND)
    return()
endif()

covise_find_package(glm)
covise_find_package(HDF5 COMPONENTS CXX)
covise_find_package(VTK COMPONENTS CommonColor CommonCore CommonDataModel IOCore IOLegacy)
using(Assimp)

SET(HEADERS
  asg.h
  asg.hpp
  linalg.hpp
  ANARIPlugin.h
  Renderer.h
  Projection.h
)
SET(SOURCES
  asg.cpp
  ANARIPlugin.cpp
  Renderer.cpp
  Projection.cpp
)

cover_add_plugin(ANARIPlugin ${HEADERS} ${SOURCES})
target_compile_definitions(ANARIPlugin PUBLIC -DASG_USE_VISIONARAY_FOR_PICKING=0)
target_link_libraries(ANARIPlugin anari::anari HDF5::HDF5 ${VTK_LIBRARIES})
if(TARGET glm::glm)
    target_link_libraries(ANARIPlugin glm::glm)
endif()
if (assimp_FOUND)
  target_compile_definitions(ANARIPlugin PUBLIC ASG_HAVE_ASSIMP=1)
endif()
if (HDF5_FOUND)
  target_compile_definitions(ANARIPlugin PUBLIC HAVE_HDF5=1)
endif()
if (VTK_FOUND)
  target_sources(ANARIPlugin PRIVATE readVTK.cpp)
  vtk_module_autoinit(
    TARGETS ANARIPlugin
    MODULES ${VTK_LIBRARIES}
  )
  target_compile_definitions(ANARIPlugin PUBLIC HAVE_VTK=1)
endif()
SET_TARGET_PROPERTIES(ANARIPlugin PROPERTIES OUTPUT_NAME ANARI)

