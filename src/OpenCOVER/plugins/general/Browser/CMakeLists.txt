#OpenCOVER plugins general Browser based on Chromium Embedded Framework
USING(CEF)

SET(HEADERS
  CEF.h
  CEFkeyssymdef.h
)

SET(SOURCES
  CEF.cpp
)

cover_add_plugin(Browser)
