USING(GDAL)

SET(HEADERS
  UrbanTempo.h
)
SET(SOURCES
  UrbanTempo.cpp
)

IF(CURL_FOUND)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${CURL_LIBRARIES})
  cover_add_plugin(UrbanTempo ${HEADERS} ${SOURCES})
  message(${PROJECT_SOURCE_DIR}/src/3rdparty/rapidjson/include)
  target_include_directories(UrbanTempo PRIVATE ${PROJECT_SOURCE_DIR}/../../src/3rdparty/visionaray/src/3rdparty/rapidjson/include ${CURL_INCLUDE_DIR})
  message("CURL_INCLUDE_DIR " ${CURL_INCLUDE_DIR})
ENDIF()
