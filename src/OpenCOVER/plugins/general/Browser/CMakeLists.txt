#OpenCOVER plugins general Browser based on Chromium Embedded Framework
USING(CEF)

if(APPLE)
  add_definitions(-DCEF18)
endif()

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
)
if(WIN32)
  SET(SOURCES
    ${SOURCES}
    CEFBrowserHelper.manifest
  )
  ADD_COVISE_EXECUTABLE(CEFBrowserHelper WIN32 ${SOURCES})
else()
  ADD_COVISE_EXECUTABLE(CEFBrowserHelper ${SOURCES})
endif()
COVISE_WNOERROR(CEFBrowserHelper)

TARGET_LINK_LIBRARIES(CEFBrowserHelper ${EXTRA_LIBS} )
