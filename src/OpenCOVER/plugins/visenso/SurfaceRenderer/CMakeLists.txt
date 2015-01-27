set(PythonLibs_FIND_VERSION 3.0)
set(Python_ADDITIONAL_VERSIONS 3.4 3.5)
set(saved_prefix_path ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH "$ENV{EXTERNLIBS}/python" ${CMAKE_PREFIX_PATH})
COVISE_FIND_PACKAGE(PythonLibs)
IF(UNIX)
 SET(PYTHON_LIBRARIES ${PYTHON_LIBRARIES} -lutil)
ENDIF(UNIX)


IF(NOT PYTHONLIBS_FOUND)
  #MESSAGE("PYTHONLIBS not found! Skipping SurfaceRenderer")
  RETURN()
ENDIF()

INCLUDE_DIRECTORIES(
  ${PYTHON_INCLUDE_DIRS}
)

IF(MSVC)
  # arrrgh pyconfig.h contains a #pragma comment(lib,"python27_d.lib")
  GET_FILENAME_COMPONENT(PYTHON_LIB_DIR "${PYTHON_DEBUG_LIBRARY}" PATH)
  LINK_DIRECTORIES(${PYTHON_LIB_DIR})
ENDIF(MSVC)

SET(HEADERS
MobiusStrip.h
ParamSurface.h
SurfaceRenderer.h
SurfaceSphere.h
)

SET(SOURCES
MobiusStrip.cpp
ParamSurface.cpp
Sphere.cpp
SurfaceRenderer.cpp
SurfaceSphere.cpp
)
cover_add_plugin(SurfaceRenderer ${PYTHON_LIBRARIES})
