#OpenCOVER plugin to load IFC files

USING(IFCPP)

SET(HEADERS
  IFCPlugin.h
  IfcPlusPlusSystem.h
)
SET(MOC_HEADERS
  IfcPlusPlusSystem.h
)

SET(SOURCES
  IFCPlugin.cpp
  IfcPlusPlusSystem.cpp
)

QT_WRAP_CPP(MOC_SRCS ${MOC_HEADERS})

LIST(APPEND SOURCES ${MOC_SRCS})

cover_add_plugin(IFC ${HEADERS})
COVISE_WNOERROR(IFC)
