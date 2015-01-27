# OpenCOVER plugins general Annotation

USING(Virvo)

# INCLUDE_DIRECTORIES( )

SET(HEADERS
  AnnotationPlugin.h
  Annotation.h
  AnnotationSensor.h
)

SET(SOURCES
  AnnotationPlugin.cpp
  Annotation.cpp 
  AnnotationSensor.cpp
)

cover_add_plugin(Annotation)

