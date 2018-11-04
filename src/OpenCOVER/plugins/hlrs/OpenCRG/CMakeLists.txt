USING(OpenCRG)
using(VehicleUtil)

SET(HEADERS
  OpenCRGPlugin.h
)
SET(SOURCES
  OpenCRGPlugin.cpp
)
cover_add_plugin(OpenCRGPlugin ${HEADERS} ${SOURCES})
set_target_properties(OpenCRGPlugin PROPERTIES OUTPUT_NAME "OpenCRG")
