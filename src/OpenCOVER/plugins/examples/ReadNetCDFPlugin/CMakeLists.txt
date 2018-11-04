SET(HEADERS
  ReadNetCDFPlugin.h
)
SET(SOURCES
  ReadNetCDFPlugin.cpp
)
cover_add_plugin(ReadNetCDFPlugin ${HEADERS} ${SOURCES})
set_target_properties(ReadNetCDFPlugin PROPERTIES OUTPUT_NAME "ReadNetCDF")
