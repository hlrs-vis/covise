USING(GDAL)

SET(HEADERS
  UrbanTempo.h
)
SET(SOURCES
  UrbanTempo.cpp
)

FIND_PACKAGE(CURL QUIET)
IF(CURL_FOUND)
  ADD_DEFINITIONS(-DHAVE_LIBCURL)
  IF(WIN32)
    ADD_DEFINITIONS(-DCURL_STATICLIB)
  ENDIF(WIN32)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${CURL_LIBRARIES})
  cover_add_plugin(UrbanTempo ${HEADERS} ${SOURCES})
  message(${PROJECT_SOURCE_DIR}/src/3rdparty/rapidjson/include)
  target_include_directories(UrbanTempo PRIVATE ${PROJECT_SOURCE_DIR}/../../src/3rdparty/visionaray/src/3rdparty/rapidjson/include ${CURL_INCLUDE_DIR})
  message("CURL_INCLUDE_DIR " ${CURL_INCLUDE_DIR})
ENDIF()