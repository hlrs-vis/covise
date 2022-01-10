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


SET(HEADERS
)
SET(SOURCES
  CEFBrowserHelper.cpp
  CEFBrowserHelper.manifest
)
if(WIN32)
ADD_COVISE_EXECUTABLE(CEFBrowserHelper WIN32)
else(WIN32)
ADD_COVISE_EXECUTABLE(CEFBrowserHelper)
endif(WIN32)
COVISE_WNOERROR(CEFBrowserHelper)

TARGET_LINK_LIBRARIES(CEFBrowserHelper ${EXTRA_LIBS} )
