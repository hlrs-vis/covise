#OpenCOVER plugins general PLMXML
USING(XERCESC)



SET(HEADERS
  PLMXMLParser.h
  PLMXMLPlugin.h
  PLMXMLSimVisitor.h
)

SET(SOURCES
  PLMXMLParser.cpp
  PLMXMLPlugin.cpp
  PLMXMLSimVisitor.cpp
)

cover_add_plugin(PLMXML)

IF(MSVC80 OR MSVC90)
  ADD_COVISE_COMPILE_FLAGS(PLMXML "/Zc:wchar_t-")
ENDIF()
