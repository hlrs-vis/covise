#set(VISIONARAY_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../3rdparty/visionaray/include")
#include_directories(${VISIONARAY_INCLUDE_DIR})

USING(VISIONARAY)

SET(HEADERS
  PointRayTracerGlobals.h
  PointRayTracerPlugin.h
  PointRayTracerDrawable.h
  PointReader.h
)
SET(SOURCES
  PointRayTracerPlugin.cpp
  PointRayTracerDrawable.cpp
  PointReader.cpp  
)

cover_add_plugin(PointRayTracerPlugin ${HEADERS} ${SOURCES})
SET_TARGET_PROPERTIES(PointRayTracerPlugin PROPERTIES OUTPUT_NAME PointRayTracer)

